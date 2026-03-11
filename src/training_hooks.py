from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class UniversalFlopCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.flops = 0

    def _get_numel(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.numel()
        elif isinstance(obj, (list, tuple)):
            return sum(self._get_numel(x) for x in obj)
        return 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        out = func(*args, **kwargs)

        func_str = str(func).lower()
        out_elements = self._get_numel(out)
        in_elements = self._get_numel(args[0]) if len(args) > 0 else 0

        # 1. OPTIMIZER: foreach / fused
        if "foreach" in func_str:
            if "addcdiv" in func_str or "addcmul" in func_str:
                self.flops += in_elements * 3
            else:
                self.flops += in_elements

        elif "aten.addcdiv" in func_str or "aten.addcmul" in func_str:
            self.flops += in_elements * 3

        # 2. ELEMENT-WISE OPERATIONS
        elif any(
            func_str.startswith(f"aten.{op}")
            for op in [
                "add", "sub", "mul", "div",
                "exp", "log", "pow", "neg", "abs",
                "relu", "sigmoid", "tanh",
                "sqrt", "rsqrt"
            ]
        ):
            elements = out_elements if out_elements > 0 else in_elements
            self.flops += elements

        # 3. REDUCTIONS
        elif any(op in func_str for op in ["aten.sum", "aten.mean", "aten.max", "aten.min", "aten.norm", "aten.var"]):
            self.flops += in_elements
            if "aten.mean" in func_str or "aten.var" in func_str:
                self.flops += out_elements

        # 4. BASIC LINEAR ALGEBRA
        elif "aten.mm" in func_str or "aten.addmm" in func_str:
            mat1 = args[1] if "addmm" in func_str else args[0]
            mat2 = args[2] if "addmm" in func_str else args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() == 2:
                m, k = mat1.shape
                _, n = mat2.shape
                self.flops += 2 * m * n * k

        elif "aten.bmm" in func_str or "aten.baddbmm" in func_str:
            mat1 = args[1] if "baddbmm" in func_str else args[0]
            mat2 = args[2] if "baddbmm" in func_str else args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() == 3:
                b, m, k = mat1.shape
                _, _, n = mat2.shape
                self.flops += 2 * b * m * n * k

        elif "aten.matmul" in func_str:
            mat1, mat2 = args[0], args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() >= 2 and mat2.dim() >= 2:
                m, k = mat1.shape[-2], mat1.shape[-1]
                _, n = mat2.shape[-2], mat2.shape[-1]
                batch_elements = mat1.numel() // (m * k)
                self.flops += 2 * batch_elements * m * n * k

        # 5. CONVOLUTIONS
        elif "aten.convolution" in func_str or "aten.conv" in func_str:
            weight_t = args[1]
            if isinstance(weight_t, torch.Tensor):
                flops_per_element = 2 * (weight_t.numel() / weight_t.shape[0])
                self.flops += int(out_elements * flops_per_element)

        # 6. ADVANCED LINEAR ALGEBRA
        elif "aten.inverse" in func_str or "aten.linalg_inv" in func_str:
            mat = args[0]
            if isinstance(mat, torch.Tensor) and mat.dim() >= 2:
                n = mat.shape[-1]
                batch_elements = mat.numel() // (n * n)
                self.flops += batch_elements * 2 * (n ** 3)

        # 7. STANDARD LOSS BACKEND OPS
        elif "aten._log_softmax" in func_str:
            self.flops += in_elements * 3
        elif "aten.nll_loss" in func_str:
            self.flops += out_elements

        return out

@dataclass
class HookHandles:
    model_fwd_handle: Optional[Any] = None

    loss_fwd_pre_handle: Optional[Any] = None
    loss_fwd_post_handle: Optional[Any] = None

    loss_bwd_pre_handle: Optional[Any] = None
    loss_bwd_post_handle: Optional[Any] = None

    opt_step_pre_handle: Optional[Any] = None
    opt_step_post_handle: Optional[Any] = None

    def remove_all(self) -> None:
        for h in (
            self.model_fwd_handle,
            self.loss_fwd_pre_handle,
            self.loss_fwd_post_handle,
            self.loss_bwd_pre_handle,
            self.loss_bwd_post_handle,
            self.opt_step_pre_handle,
            self.opt_step_post_handle,
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
            self.handles.model_fwd_handle = None

        if loss_fn is not None:
            try:
                self.handles.loss_fwd_pre_handle = loss_fn.register_forward_pre_hook(self._hook_loss_forward_pre)
            except Exception:
                self.handles.loss_fwd_pre_handle = None

            try:
                self.handles.loss_fwd_post_handle = loss_fn.register_forward_hook(self._hook_loss_forward)
            except Exception:
                self.handles.loss_fwd_post_handle = None

            try:
                self.handles.loss_bwd_pre_handle = loss_fn.register_full_backward_pre_hook(self._hook_loss_backward_pre)
            except Exception:
                self.handles.loss_bwd_pre_handle = None

            try:
                self.handles.loss_bwd_post_handle = loss_fn.register_full_backward_hook(self._hook_loss_backward)
            except Exception:
                self.handles.loss_bwd_post_handle = None

        if optimizer is not None:
            try:
                self.handles.opt_step_pre_handle = optimizer.register_step_pre_hook(self._hook_optimizer_step_pre)
            except Exception:
                self.handles.opt_step_pre_handle = None

            try:
                self.handles.opt_step_post_handle = optimizer.register_step_post_hook(self._hook_optimizer_step_post)
            except Exception:
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
        try:
            if self._loss_forward_counter is not None:
                self._loss_forward_counter.__exit__(None, None, None)
                flop = int(self._loss_forward_counter.flops)
        except Exception:
            flop = 0

        if flop > 0:
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
        try:
            if self._loss_backward_counter is not None:
                self._loss_backward_counter.__exit__(None, None, None)
                flop = int(self._loss_backward_counter.flops)
        except Exception:
            flop = 0

        if flop > 0:
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
        try:
            if self._optimizer_counter is not None:
                self._optimizer_counter.__exit__(None, None, None)
                flop = int(self._optimizer_counter.flops)
        except Exception:
            flop = 0

        if flop > 0:
            self.tracker._optimizer_flop += flop

        self._optimizer_counter = None

        if self.enable_debug_print:
            print(f"[training_hooks] optimizer step post hook called | optimizer_flop={flop}")
