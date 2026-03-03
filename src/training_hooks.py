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
        for h in (self.model_fwd_handle, self.loss_fwd_handle, self.loss_bwd_handle, self.opt_step_handle):
            try:
                if h is not None:
                    h.remove()
            except Exception:
                pass


class TorchTrainingHooks:
    """
    Installa hook su:
    - model forward (solo per debug/marker; i FLOP del modello vengono conteggiati con TorchBackend)
    - loss_fn forward -> stima i FLOP loss e li aggiunge al batch corrente
    - loss_fn backward -> opzionale (di default non aggiunge FLOP)
    - optimizer step post hook -> opzionale (di default non aggiunge FLOP)
    """

    def __init__(self, tracker, *, enable_debug_print: bool = False):
        self.tracker = tracker
        self.enable_debug_print = enable_debug_print
        self.handles = HookHandles()

        # contatori “eventi” (facoltativi, utili per debug)
        self._loss_calls = 0
        self._opt_steps = 0

    def install(self, *, model, loss_fn=None, optimizer=None) -> None:
        # --- model forward hook (facoltativo) ---
        try:
            self.handles.model_fwd_handle = model.register_forward_hook(self._hook_model_forward)
        except Exception:
            self.handles.model_fwd_handle = None

        # --- loss forward/backward hooks (solo se loss_fn è un nn.Module) ---
        if loss_fn is not None:
            try:
                self.handles.loss_fwd_handle = loss_fn.register_forward_hook(self._hook_loss_forward)
            except Exception:
                self.handles.loss_fwd_handle = None

            # register_full_backward_hook è il più affidabile per module backward
            try:
                self.handles.loss_bwd_handle = loss_fn.register_full_backward_hook(self._hook_loss_backward)
            except Exception:
                self.handles.loss_bwd_handle = None

        # --- optimizer step post hook (PyTorch >= 2.x) ---
        if optimizer is not None:
            try:
                self.handles.opt_step_handle = optimizer.register_step_post_hook(self._hook_optimizer_step_post)
            except Exception:
                self.handles.opt_step_handle = None

    def uninstall(self) -> None:
        self.handles.remove_all()

    # ---------------- HOOKS ---------------- #

    def _hook_model_forward(self, module, inputs, output):
        if self.enable_debug_print:
            print("[TorchTrainingHooks] model forward hook called")

    def _hook_loss_forward(self, module, inputs, output):
        """
        inputs tipicamente: (preds, targets) per nn.*Loss
        output: loss tensor
        stima i FLOP della loss (forward) e li aggiunge al batch corrente.
        """
        self._loss_calls += 1

        preds = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else None
        targets = inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else None

        # stima FLOP loss forward
        flop = 0
        try:
            flop = int(self.tracker._estimate_loss_flop(module, preds, targets))
        except Exception:
            flop = 0

        if flop > 0:
            # aggiorno contatore globale loss nel tracker
            self.tracker._loss_flop += flop
            # aggiungo FLOP al batch corrente (extra)
            if hasattr(self.tracker.backend, "add_extra_flop"):
                self.tracker.backend.add_extra_flop(flop)

        if self.enable_debug_print:
            print(f"[TorchTrainingHooks] loss forward hook called (loss_flop={flop})")

    def _hook_loss_backward(self, module, grad_input, grad_output):
        """
        Hook backward della loss.
        Di default NON aggiunge FLOP (di solito conta solo forward loss).
        Si può usare per debug o, per stimare FLOP backward in futuro.
        """
        if self.enable_debug_print:
            print("[TorchTrainingHooks] loss backward hook called")

    def _hook_optimizer_step_post(self, optimizer, args, kwargs):
        """
        Hook chiamato dopo optimizer.step().
        Di default non aggiunge FLOP (l’ottimizzatore è spesso fuori scope FLOP “modello”).
        """
        self._opt_steps += 1
        if self.enable_debug_print:
            print("[TorchTrainingHooks] optimizer step post hook called")
