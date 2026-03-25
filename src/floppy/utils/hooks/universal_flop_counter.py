from __future__ import annotations
from torch.utils._python_dispatch import TorchDispatchMode
import torch


class UniversalFlopCounter(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.flops = 0
        self.bops = 0

    def _get_numel(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.numel()

        elif isinstance(obj, (list, tuple)):
            return sum(self._get_numel(x) for x in obj)

        return 0

    def _get_first_tensor(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj

        elif isinstance(obj, (list, tuple)):
            for x in obj:
                tensor = self._get_first_tensor(x)
                if tensor is not None:
                    return tensor

        elif isinstance(obj, dict):
            for x in obj.values():
                tensor = self._get_first_tensor(x)
                if tensor is not None:
                    return tensor

        return None

    def _get_effective_bit_width(self, obj) -> int:
        tensor = self._get_first_tensor(obj)
        if tensor is None or not hasattr(tensor, "dtype"):
            return 32

        dtype = tensor.dtype

        if dtype in (torch.float64, torch.int64, torch.complex128):
            return 64

        elif dtype in (torch.float32, torch.int32, torch.complex64):
            return 32

        elif dtype in (torch.float16, torch.bfloat16, torch.int16):
            return 16

        elif dtype in (torch.int8, torch.uint8, torch.qint8):
            return 8

        elif str(dtype) in ("torch.quint4x2", "torch.int4"):
            return 4

        return 32

    def _add_ops(self, flop_count: int, ref_obj=None):
        flop_value = int(flop_count)
        bit_width = self._get_effective_bit_width(ref_obj)
        bop_value = int(flop_value * bit_width)
        self.flops += flop_value
        self.bops += bop_value

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        func_str = str(func).lower()
        out_elements = self._get_numel(out)
        in_elements = self._get_numel(args[0]) if len(args) > 0 else 0

        # 1. OPTIMIZER: foreach
        if "foreach" in func_str:
            if "addcdiv" in func_str or "addcmul" in func_str or "lerp" in func_str:
                self._add_ops(in_elements * 3, args)
            else:
                self._add_ops(in_elements, args)

        # 2. COMPOSITE OPERATIONS
        elif "aten.addcdiv" in func_str or "aten.addcmul" in func_str or "aten.lerp" in func_str:
            self._add_ops(in_elements * 3, args)

        # 3. ELEMENT-WISE OPERATIONS
        elif any(func_str.startswith(f"aten.{op}") for op in [
            "add", "sub", "mul", "div",
            "exp", "log", "pow", "neg", "abs",
            "relu", "sigmoid", "tanh",
            "sqrt", "rsqrt"
        ]):
            elements = out_elements if out_elements > 0 else in_elements
            self._add_ops(elements, args)

        # 4. BASIC LINEAR ALGEBRA
        elif "aten.mm" in func_str or "aten.addmm" in func_str:
            mat1 = args[1] if "addmm" in func_str else args[0]
            mat2 = args[2] if "addmm" in func_str else args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() == 2:
                m, k = mat1.shape
                _, n = mat2.shape
                self._add_ops(2 * m * n * k, (mat1, mat2))

        elif "aten.bmm" in func_str or "aten.baddbmm" in func_str:
            mat1 = args[1] if "baddbmm" in func_str else args[0]
            mat2 = args[2] if "baddbmm" in func_str else args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() == 3:
                b, m, k = mat1.shape
                _, _, n = mat2.shape
                self._add_ops(2 * b * m * n * k, (mat1, mat2))

        elif "aten.matmul" in func_str:
            mat1, mat2 = args[0], args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() >= 2 and mat2.dim() >= 2:
                m, k = mat1.shape[-2], mat1.shape[-1]
                _, n = mat2.shape[-2], mat2.shape[-1]
                batch_elements = mat1.numel() // (m * k)
                self._add_ops(2 * batch_elements * m * n * k, (mat1, mat2))

        # 5. CONVOLUTIONS
        elif "aten.convolution" in func_str or "aten.conv" in func_str:
            weight_t = args[1]
            if isinstance(weight_t, torch.Tensor):
                flops_per_element = 2 * (weight_t.numel() / weight_t.shape[0])
                self._add_ops(int(out_elements * flops_per_element), weight_t)

        # 6. ADVANCED LINEAR ALGEBRA
        elif "aten.inverse" in func_str or "aten.linalg_inv" in func_str:
            mat = args[0]
            if isinstance(mat, torch.Tensor) and mat.dim() >= 2:
                n = mat.shape[-1]
                batch_elements = mat.numel() // (n * n)
                self._add_ops(batch_elements * 2 * (n ** 3), mat)

        # 7. STANDARD LOSS BACKEND OPS
        elif "aten._log_softmax" in func_str:
            self._add_ops(in_elements * 3, args)

        elif "aten.nll_loss" in func_str:
            self._add_ops(out_elements, args)

        return out
