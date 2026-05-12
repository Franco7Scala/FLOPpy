from __future__ import annotations
from typing import Optional
from .hook_handles import HookHandles
from .universal_flop_counter import UniversalFlopCounter
from ....utils.utility import Color, cprint


class TorchTrainingHooks:

    def __init__(self, tracker, *, enable_debug_print: bool = False):
        self.tracker = tracker
        self.enable_debug_print = enable_debug_print
        self.handles = HookHandles()
        self.model_forward_calls = 0
        self.loss_forward_calls = 0
        self.loss_backward_calls = 0
        self.optimizer_step_calls = 0
        self._loss_forward_counter: Optional[UniversalFlopCounter] = None
        self._optimizer_counter: Optional[UniversalFlopCounter] = None
        self._loss_forward_active = False
        self._optimizer_active = False

    # ------------------------------------------------------------
    # API
    # ------------------------------------------------------------
    def install(self, *, model, loss_fn=None, optimizer=None) -> None:
        try:
            self.handles.model_fwd_handle = model.register_forward_hook(self._hook_model_forward)
        except Exception:
            cprint("[training_hooks] Warning: model forward hook registration failed. Model forward FLOP/BOP will not be counted.", Color.WARNING)
            self.handles.model_fwd_handle = None

        if loss_fn is not None:
            try:
                self.handles.loss_fwd_pre_handle = loss_fn.register_forward_pre_hook(self._hook_loss_forward_pre)

            except Exception:
                cprint("[training_hooks] Warning: loss forward pre-hook registration failed. Loss forward FLOP/BOP will not be counted.", Color.WARNING)
                self.handles.loss_fwd_pre_handle = None

            try:
                self.handles.loss_fwd_post_handle = loss_fn.register_forward_hook(self._hook_loss_forward)

            except Exception:
                cprint("[training_hooks] Warning: loss forward post-hook registration failed. Loss forward FLOP/BOP will not be counted.", Color.WARNING)
                self.handles.loss_fwd_post_handle = None

        if optimizer is not None:
            try:
                self.handles.opt_step_pre_handle = optimizer.register_step_pre_hook(self._hook_optimizer_step_pre)

            except Exception:
                cprint("[training_hooks] Warning: optimizer step pre-hook registration failed. Optimizer FLOP/BOP will not be counted.", Color.WARNING)
                self.handles.opt_step_pre_handle = None

            try:
                self.handles.opt_step_post_handle = optimizer.register_step_post_hook(self._hook_optimizer_step_post)

            except Exception:
                cprint("[training_hooks] Warning: optimizer step post-hook registration failed. Optimizer FLOP/BOP will not be counted.", Color.WARNING)
                self.handles.opt_step_post_handle = None

    def uninstall(self) -> None:
        self.handles.remove_all()

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------
    def _hook_model_forward(self, module, inputs, output) -> None:
        self.model_forward_calls += 1
        self.tracker._last_model_output = output

        if self.enable_debug_print:
            print("[training_hooks] model forward hook called")

    # ------------------------------------------------------------
    # LOSS FORWARD
    # ------------------------------------------------------------
    def _hook_loss_forward_pre(self, module, inputs) -> None:
        self._loss_forward_counter = UniversalFlopCounter()
        self._loss_forward_active = False
        try:
            self._loss_forward_counter.__enter__()
            self._loss_forward_active = True

        except Exception:
            self._loss_forward_counter = None
            self._loss_forward_active = False

    def _hook_loss_forward(self, module, inputs, output) -> None:
        self.loss_forward_calls += 1
        flop = 0
        bop = 0
        try:
            if self._loss_forward_counter is not None and self._loss_forward_active:
                self._loss_forward_counter.__exit__(None, None, None)
                flop = int(getattr(self._loss_forward_counter, "flops", 0))
                bop = int(getattr(self._loss_forward_counter, "bops", 0))

        except Exception:
            flop = 0
            bop = 0

        self.tracker._loss_forward_flop += flop
        self.tracker._loss_forward_bop += bop
        self._loss_forward_counter = None
        self._loss_forward_active = False
        if self.enable_debug_print:
            print(f"[training_hooks] loss forward hook called | loss_forward_flop={flop} | loss_forward_bop={bop}")

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------
    def _hook_optimizer_step_pre(self, optimizer, args, kwargs) -> None:
        self._optimizer_counter = UniversalFlopCounter()
        self._optimizer_active = False
        try:
            self._optimizer_counter.__enter__()
            self._optimizer_active = True

        except Exception:
            self._optimizer_counter = None
            self._optimizer_active = False

    def _hook_optimizer_step_post(self, optimizer, args, kwargs) -> None:
        self.optimizer_step_calls += 1
        self.tracker._last_optimizer_step_seen = True
        flop = 0
        bop = 0
        # 1. Try to get FLOPs from the standard ATen Dispatcher
        try:
            if self._optimizer_counter is not None and self._optimizer_active:
                self._optimizer_counter.__exit__(None, None, None)
                flop = int(getattr(self._optimizer_counter, "flops", 0))
                bop = int(getattr(self._optimizer_counter, "bops", 0))

        except Exception:
            flop = 0
            bop = 0

        # ------------------------------------------------------------
        # 2. ESCAPE HATCH FOR CUSTOM / FUSED OPTIMIZERS
        # ------------------------------------------------------------
        # If the dispatcher missed the operations (flop == 0), check if it's a known C++/CUDA custom optimizer
        if flop == 0:
            opt_name = optimizer.__class__.__name__
            # List of the most common fused/quantized optimizers that bypass PyTorch standard ops
            custom_optimizers = [
                "Adam8bit", "AdamW8bit", "PagedAdam8bit", "PagedAdamW8bit",  # BitsAndBytes (QLoRA)
                "FusedAdam", "FusedAdamW", "FusedSGD",  # NVIDIA Apex / Megatron
                "DeepSpeedCPUAdam", "DeepSpeedZeroOptimizer",  # DeepSpeed
                "Lion8bit", "PagedLion8bit"  # Emerging custom optimizers
            ]
            # If the current optimizer matches any of the custom names
            if any(known_opt in opt_name for known_opt in custom_optimizers):
                # A. Count the total number of trainable parameters handled by this optimizer
                num_params = sum(
                    p.numel()
                    for group in optimizer.param_groups
                    for p in group["params"]
                    if p.requires_grad
                )
                # B. Algorithmic FLOP estimation per parameter
                # Adam variants do ~10 FLOPs per param (momentum, variance, bias correction, weight update)
                # SGD variants do ~5 FLOPs per param (velocity, weight update)
                ops_per_param = 5 if "SGD" in opt_name else 10
                flop = num_params * ops_per_param
                # C. BOPs estimation based on the optimizer's bit-width storage
                # If "8bit" is in the name, it stores states in INT8, otherwise assume standard FP16/BF16
                bit_width = 8 if "8bit" in opt_name.lower() else 16
                bop = flop * bit_width

        # 3. Update the global tracker states
        self.tracker._optimizer_flop += flop
        self.tracker._optimizer_bop += bop
        self._optimizer_counter = None
        self._optimizer_active = False
        if self.enable_debug_print:
            print(f"[training_hooks] optimizer step post hook called | optimizer_flop={flop} | optimizer_bop={bop}")
