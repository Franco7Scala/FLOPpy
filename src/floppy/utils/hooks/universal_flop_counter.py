from __future__ import annotations
from torch.utils._python_dispatch import TorchDispatchMode

import torch


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

        # 1. OPTIMIZER: foreach
        if "foreach" in func_str:
            if "addcdiv" in func_str or "addcmul" in func_str or "lerp" in func_str:
                self.flops += in_elements * 3

            else:
                self.flops += in_elements

        # 2. COMPOSITE OPERATIONS
        elif "aten.addcdiv" in func_str or "aten.addcmul" in func_str or "aten.lerp" in func_str:
            self.flops += in_elements * 3

        # 3. ELEMENT-WISE OPERATIONS
        elif any(func_str.startswith(f"aten.{op}") for op in [
                    "add", "sub", "mul", "div",
                    "exp", "log", "pow", "neg", "abs",
                    "relu", "sigmoid", "tanh",
                    "sqrt", "rsqrt"]):
            elements = out_elements if out_elements > 0 else in_elements
            self.flops += elements

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
