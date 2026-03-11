from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class HookHandles:
    model_fwd_handle: Optional[Any] = None
    loss_fwd_handle: Optional[Any] = None
    loss_bwd_handle: Optional[Any] = None
    opt_step_handle: Optional[Any] = None

    def remove_all(self) -> None:
        for h in (
            self.model_fwd_handle,
            self.loss_fwd_handle,
            self.loss_bwd_handle,
            self.opt_step_handle,
        ):
            try:
                if h is not None:
                    h.remove()
            except Exception:
                pass

class TorchTrainingHooks:
    def __init__(self, tracker, *, enable_debug_print: bool = False):
        self.tracker = tracker
        self.enable_debug_print = enable_debug_print
        self.handles = HookHandles()

        self.model_forward_calls = 0
        self.loss_forward_calls = 0
        self.loss_backward_calls = 0
        self.optimizer_step_calls = 0

        self._last_loss_module = None
        self._last_loss_preds = None
        self._last_loss_targets = None

    def install(self, *, model, loss_fn=None, optimizer=None) -> None:
        try:
            self.handles.model_fwd_handle = model.register_forward_hook(self._hook_model_forward)
        except Exception:
            self.handles.model_fwd_handle = None

        if loss_fn is not None:
            try:
                self.handles.loss_fwd_handle = loss_fn.register_forward_hook(self._hook_loss_forward)
            except Exception:
                self.handles.loss_fwd_handle = None

            try:
                self.handles.loss_bwd_handle = loss_fn.register_full_backward_hook(self._hook_loss_backward)
            except Exception:
                self.handles.loss_bwd_handle = None

        if optimizer is not None:
            try:
                self.handles.opt_step_handle = optimizer.register_step_post_hook(self._hook_optimizer_step_post)
            except Exception:
                self.handles.opt_step_handle = None

    def uninstall(self) -> None:
        self.handles.remove_all()

    # ------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------

    def _hook_model_forward(self, module, inputs, output) -> None:
        self.model_forward_calls += 1
        self.tracker._last_model_output = output

        if self.enable_debug_print:
            print("[training_hooks] model forward hook called")

    def _hook_loss_forward(self, module, inputs, output) -> None:
        self.loss_forward_calls += 1

        preds = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else None
        targets = inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else None

        self._last_loss_module = module
        self._last_loss_preds = preds
        self._last_loss_targets = targets

        flop = 0
        try:
            flop = int(
                self.tracker._estimate_loss_flop(
                    loss=module,
                    outputs=preds,
                    targets=targets,
                    extra=None,
                )
            )
        except Exception:
            flop = 0

        if flop > 0:
            self.tracker._loss_forward_flop += flop

        if self.enable_debug_print:
            print(f"[training_hooks] loss forward hook called | loss_flop={flop}")

    def _hook_loss_backward(self, module, grad_input, grad_output) -> None:
        self.loss_backward_calls += 1
        self.tracker._last_backward_seen = True

        flop = 0
        try:
            flop = int(
                self.tracker._estimate_loss_backward_flop(
                    loss=self._last_loss_module,
                    outputs=self._last_loss_preds,
                    targets=self._last_loss_targets,
                    extra=None,
                )
            )
        except Exception:
            flop = 0

        if flop > 0:
            self.tracker._loss_backward_flop += flop

        if self.enable_debug_print:
            print(f"[training_hooks] loss backward hook called | loss_bwd_flop={flop}")

    def _hook_optimizer_step_post(self, optimizer, args, kwargs) -> None:
        self.optimizer_step_calls += 1
        self.tracker._last_optimizer_step_seen = True

        flop = 0
        try:
            flop = int(self.tracker._estimate_optimizer_flop(optimizer))
        except Exception:
            flop = 0

        if flop > 0:
            self.tracker._optimizer_flop += flop

        if self.enable_debug_print:
            print(f"[training_hooks] optimizer step post hook called | opt_flop={flop}")
