from __future__ import annotations
from typing import Optional
from torch.utils._python_dispatch import TorchDispatchMode
from floppy.utils.hooks.hook_handles import HookHandles
from floppy.utils.hooks.universal_flop_counter import UniversalFlopCounter
from floppy.utils.utility import Color, cprint

import torch


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
        self._loss_backward_counter: Optional[UniversalFlopCounter] = None
        self._optimizer_counter: Optional[UniversalFlopCounter] = None

    # ------------------------------------------------------------
    # API
    # ------------------------------------------------------------
    def install(self, *, model, loss_fn=None, optimizer=None) -> None:
        try:
            self.handles.model_fwd_handle = model.register_forward_hook(self._hook_model_forward)

        except Exception:
            cprint("[training_hooks] Warning: model forward hook registration failed. Model forward FLOP will not be counted.", Color.WARNING)
            self.handles.model_fwd_handle = None

        if loss_fn is not None:
            try:
                self.handles.loss_fwd_pre_handle = loss_fn.register_forward_pre_hook(self._hook_loss_forward_pre)

            except Exception:
                cprint("[training_hooks] Warning: loss forward pre-hook registration failed. Loss forward FLOP will not be counted.", Color.WARNING)
                self.handles.loss_fwd_pre_handle = None

            try:
                self.handles.loss_fwd_post_handle = loss_fn.register_forward_hook(self._hook_loss_forward)

            except Exception:
                cprint("[training_hooks] Warning: loss forward post-hook registration failed. Loss forward FLOP will not be counted.", Color.WARNING)
                self.handles.loss_fwd_post_handle = None

            try:
                self.handles.loss_bwd_pre_handle = loss_fn.register_full_backward_pre_hook(self._hook_loss_backward_pre)

            except Exception:
                cprint("[training_hooks] Warning: loss backward pre-hook registration failed. Loss backward FLOP will not be counted.", Color.WARNING)
                self.handles.loss_bwd_pre_handle = None

            try:
                self.handles.loss_bwd_post_handle = loss_fn.register_full_backward_hook(self._hook_loss_backward)

            except Exception:
                cprint("[training_hooks] Warning: loss backward post-hook registration failed. Loss backward FLOP will not be counted.", Color.WARNING)
                self.handles.loss_bwd_post_handle = None

        if optimizer is not None:
            try:
                self.handles.opt_step_pre_handle = optimizer.register_step_pre_hook(self._hook_optimizer_step_pre)

            except Exception:
                cprint("[training_hooks] Warning: optimizer step pre-hook registration failed. Optimizer FLOP will not be counted.", Color.WARNING)
                self.handles.opt_step_pre_handle = None

            try:
                self.handles.opt_step_post_handle = optimizer.register_step_post_hook(self._hook_optimizer_step_post)

            except Exception:
                cprint("[training_hooks] Warning: optimizer step post-hook registration failed. Optimizer FLOP will not be counted.", Color.WARNING)
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
        self._loss_forward_counter.__enter__()

    def _hook_loss_forward(self, module, inputs, output) -> None:
        self.loss_forward_calls += 1
        flop = 0
        if self._loss_forward_counter is not None:
            self._loss_forward_counter.__exit__(None, None, None)
            flop = int(self._loss_forward_counter.flops)

        self.tracker._loss_forward_flop += flop
        self._loss_forward_counter = None
        if self.enable_debug_print:
            print(f"[training_hooks] loss forward hook called | loss_forward_flop={flop}")

    # ------------------------------------------------------------
    # LOSS BACKWARD
    # ------------------------------------------------------------
    def _hook_loss_backward_pre(self, module, grad_output) -> None:
        self._loss_backward_counter = UniversalFlopCounter()
        self._loss_backward_counter.__enter__()

    def _hook_loss_backward(self, module, grad_input, grad_output) -> None:
        self.loss_backward_calls += 1
        self.tracker._last_backward_seen = True
        flop = 0
        if self._loss_backward_counter is not None:
            self._loss_backward_counter.__exit__(None, None, None)
            flop = int(self._loss_backward_counter.flops)

        self.tracker._loss_backward_flop += flop
        self._loss_backward_counter = None
        if self.enable_debug_print:
            print(f"[training_hooks] loss backward hook called | loss_backward_flop={flop}")

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------
    def _hook_optimizer_step_pre(self, optimizer, args, kwargs) -> None:
        self._optimizer_counter = UniversalFlopCounter()
        self._optimizer_counter.__enter__()

    def _hook_optimizer_step_post(self, optimizer, args, kwargs) -> None:
        self.optimizer_step_calls += 1
        self.tracker._last_optimizer_step_seen = True
        flop = 0
        if self._optimizer_counter is not None:
            self._optimizer_counter.__exit__(None, None, None)
            flop = int(self._optimizer_counter.flops)

        self.tracker._optimizer_flop += flop
        self._optimizer_counter = None
        if self.enable_debug_print:
            print(f"[training_hooks] optimizer step post hook called | optimizer_flop={flop}")
