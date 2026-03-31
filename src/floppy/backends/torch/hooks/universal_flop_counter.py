import torch
import math

from torch.utils._python_dispatch import TorchDispatchMode


class UniversalFlopCounter(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.flops = 0
        self.bops = 0
        self.paused = False

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

    def _add_ops(self, flop_count: int, ref_obj=None, op_name: str = "unknown"):
        flop_value = int(flop_count)
        bit_width = self._get_effective_bit_width(ref_obj)
        self.flops += flop_value
        self.bops += int(flop_value * bit_width)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.paused:
            kwargs = kwargs or {}
            return func(*args, **kwargs)

        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        func_str = str(func).lower()

        out_elements = self._get_numel(out)
        in_elements = self._get_numel(args[0]) if len(args) > 0 else 0

        # 1. ELEMENT-WISE
        base_op = func_str.split(".")[1] if "." in func_str else func_str.replace("aten.", "")
        if base_op in [
            "add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_",
            "exp", "exp_", "log", "log_", "pow", "pow_", "neg", "neg_", "abs", "abs_",
            "relu", "relu_", "sigmoid", "sigmoid_", "tanh", "tanh_",
            "sqrt", "sqrt_", "rsqrt", "rsqrt_", "gelu", "gelu_", "silu", "silu_", "mish", "mish_",
            "_foreach_add", "_foreach_add_", "_foreach_sub", "_foreach_sub_",
            "_foreach_mul", "_foreach_mul_", "_foreach_div", "_foreach_div_",
            "_foreach_sqrt", "_foreach_sqrt_", "_foreach_exp", "_foreach_exp_",
            "_foreach_neg", "_foreach_neg_"]:
            elements = out_elements if out_elements > 0 else in_elements
            self._add_ops(elements, args, func_str)

        # 1.5 COMPLEX OPTIMIZER OPS
        elif base_op in [
            "addcdiv", "addcdiv_", "_foreach_addcdiv", "_foreach_addcdiv_",
            "addcmul", "addcmul_", "_foreach_addcmul", "_foreach_addcmul_"]:
            elements = out_elements if out_elements > 0 else in_elements
            self._add_ops(elements * 3, args, func_str)

        # 2. CORE LINEAR ALGEBRA
        elif "aten.mm" in func_str or "aten.addmm" in func_str:
            mat1 = args[1] if "addmm" in func_str else args[0]
            mat2 = args[2] if "addmm" in func_str else args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() == 2:
                m, k = mat1.shape
                _, n = mat2.shape
                self._add_ops(2 * m * n * k, mat1, func_str)

        elif "aten.bmm" in func_str or "aten.baddbmm" in func_str:
            mat1 = args[1] if "baddbmm" in func_str else args[0]
            mat2 = args[2] if "baddbmm" in func_str else args[1]
            if isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor) and mat1.dim() == 3:
                b, m, k = mat1.shape
                _, _, n = mat2.shape
                self._add_ops(2 * b * m * n * k, mat1, func_str)

        # 3. EMBEDDINGS E NORMS
        elif "aten.embedding" in func_str:
            weight, indices = args[0], args[1]
            if isinstance(weight, torch.Tensor) and isinstance(indices, torch.Tensor):
                self._add_ops(indices.numel() * weight.shape[-1], weight, func_str)

        elif "aten.layer_norm" in func_str or "aten.native_layer_norm" in func_str or "aten.rms_norm" in func_str:
            x = args[0]
            if isinstance(x, torch.Tensor):
                self._add_ops(4 * x.numel(), x, func_str)

        # 4. SOFTMAX E LOSS FUNCTIONS
        elif "aten._softmax" in func_str or "aten.softmax" in func_str or "aten._log_softmax" in func_str:
            self._add_ops(in_elements * 3, args, func_str)

        elif "aten.nll_loss" in func_str or "aten.cross_entropy_loss" in func_str:
            self._add_ops(in_elements, args, func_str)

        # 5. CONVOLUTION (Per Vision Transformers, ResNet, CNNs)
        elif "aten.convolution" in func_str or "aten.conv" in func_str:
            weight_t = args[1]
            if isinstance(weight_t, torch.Tensor):
                flops_per_element = 2 * (weight_t.numel() / weight_t.shape[0])
                self._add_ops(int(out_elements * flops_per_element), weight_t, func_str)

        # 6. FLASH ATTENTION / SDPA
        elif "aten._scaled_dot_product" in func_str:
            q, k, v = args[0], args[1], args[2]
            if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                # 1. Q @ K^T -> 2 * seq_len * seq_len * head_dim
                # 2. Softmax -> 3 * seq_len * seq_len
                # 3. Attn @ V -> 2 * seq_len * seq_len * head_dim
                batch_size = q.shape[0]
                num_heads = q.shape[1]
                seq_len_q = q.shape[2]
                seq_len_k = k.shape[2]
                head_dim = q.shape[3]
                flops_qk = 2 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
                flops_softmax = 3 * batch_size * num_heads * seq_len_q * seq_len_k
                flops_v = 2 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
                self._add_ops(flops_qk + flops_softmax + flops_v, q, func_str)

        # 7. GRAPH NEURAL NETWORKS (PyTorch Geometric & Routing)
        elif any(op in func_str for op in [
            "aten.scatter", "aten.scatter_add", "aten.scatter_reduce",
            "aten.index_add", "aten.index_select", "aten.gather"
        ]):
            self._add_ops(out_elements, args, func_str)

        elif "aten.bincount" in func_str:
            self._add_ops(in_elements, args, func_str)

        # 8. POOLING E UPSAMPLING (Computer Vision / CNN)
        elif any(op in func_str for op in ["aten.max_pool", "aten.avg_pool", "aten.adaptive_avg_pool"]):
            self._add_ops(out_elements, args, func_str)

        elif "aten.upsample" in func_str or "aten._upsample" in func_str:
            self._add_ops(out_elements * 4, args, func_str)

        # 9. MIXTURE OF EXPERTS (MoE Routers & Sorting)
        elif "aten.topk" in func_str or "aten.sort" in func_str or "aten.argsort" in func_str:
            if in_elements > 0:
                self._add_ops(int(in_elements * math.log2(max(2, in_elements))), args, func_str)

        # 10. RECURRENT NNs (LSTM / GRU / RNN)
        elif "aten.lstm" in func_str or "aten.gru" in func_str or "aten.rnn" in func_str:
            x = args[0]
            if isinstance(x, torch.Tensor) and x.dim() >= 2:
                self._add_ops(x.numel() * 8, args, func_str)

        return out
