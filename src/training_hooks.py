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
    """
    Hook manager per training PyTorch.

    Obiettivi:
    - contare i FLOP della loss nel suo forward
    - osservare backward della loss e step dell'optimizer
    - mantenere il comportamento "trasparente" per l'utente
    """

    def __init__(self, tracker, *, enable_debug_print: bool = False):
        self.tracker = tracker
        self.enable_debug_print = enable_debug_print
        self.handles = HookHandles()

        # contatori per debug / diagnostica
        self.model_forward_calls: int = 0
        self.loss_forward_calls: int = 0
        self.loss_backward_calls: int = 0
        self.optimizer_step_calls: int = 0

    # ------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------

    def install(self, *, model, loss_fn=None, optimizer=None) -> None:
        """
        Installa gli hook su modello, loss e optimizer.
        """
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
    # HOOKS
    # ------------------------------------------------------------

    def _hook_model_forward(self, module, inputs, output) -> None:
        """
        Hook chiamato dopo il forward del modello.

        Nota:
        - NON aggiunge FLOP, i FLOP del modello sono già
          conteggiati dal backend tramite i forward hook sui layer.
        - Serve a marcare il passaggio del forward e a mantenere
          uno stato coerente del training.
        """
        self.model_forward_calls += 1

        # stato  per debug/report
        if hasattr(self.tracker, "_last_model_output"):
            self.tracker._last_model_output = output
        else:
            try:
                self.tracker._last_model_output = output
            except Exception:
                pass

        if self.enable_debug_print:
            print("[training_hooks] model forward hook called")

    def _hook_loss_forward(self, module, inputs, output) -> None:
        """
        Hook chiamato dopo il forward della loss.

        inputs tipicamente:
            inputs[0] = preds / logits
            inputs[1] = targets

        output:
            tensore scalare della loss

        Stima dei FLOP della loss e aggiunti al batch corrente
        tramite backend.add_extra_flop(...).
        """
        self.loss_forward_calls += 1

        preds = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else None
        targets = inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else None

        flop = 0
        try:
            # supporta sia una firma tipo:
            # _estimate_loss_flop(loss=..., outputs=..., targets=..., extra=None)
            # sia eventuali versioni più semplici
            flop = int(
                self.tracker._estimate_loss_flop(
                    loss=module,
                    outputs=preds,
                    targets=targets,
                    extra=None,
                )
            )
        except TypeError:
            try:
                flop = int(self.tracker._estimate_loss_flop(module, preds, targets))
            except Exception:
                flop = 0
        except Exception:
            flop = 0

        if flop > 0:
            # aggiorna breakdown loss
            if hasattr(self.tracker, "_loss_flop"):
                self.tracker._loss_flop += flop

            # aggiunge i FLOP della loss al batch corrente
            if hasattr(self.tracker, "backend") and hasattr(self.tracker.backend, "add_extra_flop"):
                self.tracker.backend.add_extra_flop(flop)

        if self.enable_debug_print:
            print(f"[training_hooks] loss forward hook called | loss_flop={flop}")

    def _hook_loss_backward(self, module, grad_input, grad_output) -> None:
        """
        Hook chiamato durante il backward della loss.

        """
        self.loss_backward_calls += 1

        if hasattr(self.tracker, "_last_backward_seen"):
            self.tracker._last_backward_seen = True
        else:
            try:
                self.tracker._last_backward_seen = True
            except Exception:
                pass

        if self.enable_debug_print:
            print("[training_hooks] loss backward hook called")

    def _hook_optimizer_step_post(self, optimizer, args, kwargs) -> None:
        """
        Hook chiamato dopo optimizer.step().

        """
        self.optimizer_step_calls += 1

        if hasattr(self.tracker, "_last_optimizer_step_seen"):
            self.tracker._last_optimizer_step_seen = True
        else:
            try:
                self.tracker._last_optimizer_step_seen = True
            except Exception:
                pass

        if self.enable_debug_print:
            print("[training_hooks] optimizer step post hook called")
