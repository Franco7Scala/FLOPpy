from __future__ import annotations
from .base import BaseBackend

import threading
import torch
import torch.nn as nn


class TorchBackend(BaseBackend):
    """
    Backend for PyTorch models.

    Responsibilities:
    - handles DataParallel / DDP via .module
    - counts ONLY model FLOP

    Supported layer families:
        * Conv1d / Conv2d / Conv3d
        * ConvTranspose1d / 2d / 3d
        * Linear
        * Pooling (Max/Avg/Adaptive)
        * Normalization (BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm)
        * Activations (ReLU, LeakyReLU, PReLU, Sigmoid, Tanh)
        * Softmax family
        * RNN / LSTM / GRU
        * RNNCell / LSTMCell / GRUCell
        * MultiheadAttention
        * Embedding / EmbeddingBag
        * Transformer containers (counted through submodules)
    """

    def __init__(self, model: nn.Module, logger=None):
        self._root_model = model
        if isinstance(model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module

        super().__init__(model, logger=logger)
        self._layer_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._root_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current_forward_flop: int = 0
        self._current_forward_bop: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def start(self):
        tracked_types = (
            # Convolutions / Linear
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,

            # Pooling
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,

            # Normalization
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,

            # Activations
            nn.ReLU,
            nn.LeakyReLU,
            nn.PReLU,
            nn.Sigmoid,
            nn.Tanh,

            # Softmax family
            nn.Softmax,
            nn.Softmin,
            nn.Softmax2d,
            nn.LogSoftmax,

            # RNN family
            nn.RNN,
            nn.LSTM,
            nn.GRU,
            nn.RNNCell,
            nn.LSTMCell,
            nn.GRUCell,

            # Attention / Embeddings
            nn.MultiheadAttention,
            nn.Embedding,
            nn.EmbeddingBag,

            # Transformer high-level containers
            nn.Transformer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,

            # Wrapper
            nn.DataParallel
        )

        for module in self.model.modules():
            if isinstance(module, tracked_types):
                handle = module.register_forward_hook(self._layer_hook)
                self._layer_handles.append(handle)

        # RMSNorm (if available in current torch version)
        if hasattr(nn, "RMSNorm"):
            for module in self.model.modules():
                if isinstance(module, nn.RMSNorm):
                    handle = module.register_forward_hook(self._layer_hook)
                    self._layer_handles.append(handle)

        # root hooks to delimit one model forward
        pre_handle = self._root_model.register_forward_pre_hook(self._on_forward_start)
        post_handle = self._root_model.register_forward_hook(self._on_forward_end)
        self._root_handles.extend([pre_handle, post_handle])

    def stop(self):
        for handle in self._layer_handles:
            handle.remove()

        for handle in self._root_handles:
            handle.remove()

        self._layer_handles.clear()
        self._root_handles.clear()

    # ------------------------------------------------------------
    # Root forward hooks
    # ------------------------------------------------------------
    def _on_forward_start(self, module, inputs):
        self._current_forward_flop = 0
        self._current_forward_bop = 0

    def _on_forward_end(self, module, inputs, output):
        self._batch_idx += 1
        # FLOP
        forward_flop = int(self._current_forward_flop)
        self._last_batch_flop = forward_flop
        self.total_flop += forward_flop
        # BOP
        forward_bop = int(self._current_forward_bop)
        self._last_batch_bop = forward_bop
        self.total_bop += forward_bop

    # ------------------------------------------------------------
    # Layer hook 
    # ------------------------------------------------------------
    def _layer_hook(self, layer, inputs, output):
        x = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else None
        y = output

        if isinstance(y, (tuple, list)):
            y = next((o for o in y if isinstance(o, torch.Tensor)), None)

        flop = 0

        # --- CONV --- #
        if isinstance(layer, nn.Conv1d):
            flop = self._conv1d_flop(layer, x, y)

        elif isinstance(layer, nn.Conv2d):
            flop = self._conv2d_flop(layer, x, y)

        elif isinstance(layer, nn.Conv3d):
            flop = self._conv3d_flop(layer, x, y)

        elif isinstance(layer, nn.ConvTranspose1d):
            flop = self._convtranspose1d_flop(layer, x, y)

        elif isinstance(layer, nn.ConvTranspose2d):
            flop = self._convtranspose2d_flop(layer, x, y)

        elif isinstance(layer, nn.ConvTranspose3d):
            flop = self._convtranspose3d_flop(layer, x, y)

        # --- LINEAR --- #
        elif isinstance(layer, nn.Linear):
            flop = self._linear_flop(layer, x, y)

        # --- POOLING --- #
        elif isinstance(layer, (nn.MaxPool1d, nn.AvgPool1d, nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)):
            flop = self._pool1d_flop(layer, x, y)

        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
            flop = self._pool2d_flop(layer, x, y)

        elif isinstance(layer, (nn.MaxPool3d, nn.AvgPool3d, nn.AdaptiveAvgPool3d, nn.AdaptiveMaxPool3d)):
            flop = self._pool3d_flop(layer, x, y)

        # --- NORMALIZATION --- #
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            flop = self._batchnorm_flop(y)

        elif isinstance(layer, nn.LayerNorm):
            flop = self._layernorm_flop(y)

        elif isinstance(layer, nn.GroupNorm):
            flop = self._groupnorm_flop(y)

        elif isinstance(layer, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            flop = self._instancenorm_flop(y)

        elif hasattr(nn, "RMSNorm") and isinstance(layer, nn.RMSNorm):
            flop = self._rmsnorm_flop(layer, y)

        # --- ACTIVATIONS --- #
        elif isinstance(layer, nn.ReLU):
            flop = self._relu_flop(y)

        elif isinstance(layer, nn.LeakyReLU):
            flop = self._leakyrelu_flop(y)

        elif isinstance(layer, nn.PReLU):
            flop = self._prelu_flop(y)

        elif isinstance(layer, nn.Sigmoid):
            flop = self._sigmoid_flop(y)

        elif isinstance(layer, nn.Tanh):
            flop = self._tanh_flop(y)

        # --- SOFTMAX FAMILY --- #
        elif isinstance(layer, (nn.Softmax, nn.Softmin, nn.Softmax2d, nn.LogSoftmax)):
            flop = self._softmax_family_flop(layer, y)

        # --- RNN / LSTM / GRU --- #
        elif isinstance(layer, (nn.RNN, nn.LSTM, nn.GRU)):
            flop = self._rnn_flop(layer, x)

        # --- RNN CELLS --- #
        elif isinstance(layer, nn.RNNCell):
            flop = self._rnncell_flop(layer, inputs)

        elif isinstance(layer, nn.LSTMCell):
            flop = self._lstmcell_flop(layer, inputs)

        elif isinstance(layer, nn.GRUCell):
            flop = self._grucell_flop(layer, inputs)

        # --- MULTIHEAD ATTENTION --- #
        elif isinstance(layer, nn.MultiheadAttention):
            flop = self._mha_flop(layer, inputs)

        # --- EMBEDDING --- #
        elif isinstance(layer, nn.Embedding):
            flop = self._embedding_flop(layer, x)

        elif isinstance(layer, nn.EmbeddingBag):
            flop = self._embeddingbag_flop(layer, x)

        # --- CONTAINERS / WRAPPERS --- #
        elif isinstance(
            layer,
            (
                nn.Transformer,
                nn.TransformerEncoder,
                nn.TransformerDecoder,
                nn.TransformerEncoderLayer,
                nn.TransformerDecoderLayer,
                nn.DataParallel,
            ),
        ):
            # Container modules (structural layers):
            # These modules define the architecture structure but do not execute arithmetic operations directly. 
            # Their computational cost is already accounted for by hooks attached to their internal layers.
            flop = 0

        with self._lock:
            # Accumulate pure FLOPs (Algorithmic complexity)
            self._current_forward_flop += int(flop)
            # Extract the effective bit-width, accounting for packed/quantized tensors
            bit_width = self._get_effective_bit_width(layer, x)
            # Calculate hardware computational effort (Bit-Operations or BOPs)
            bop = int(flop) * bit_width
            self._current_forward_bop += bop

    def _get_effective_bit_width(self, layer: nn.Module, tensor: torch.Tensor) -> int:
        """
        Determines the effective bit-width of the operations.
        It inspects both the tensor dtype and specific layer attributes to
        correctly identify "packed" quantized tensors (e.g., INT4, INT2)
        from libraries like bitsandbytes, AutoGPTQ, or AWQ.
        """
        # 1. Check via layer class name (e.g., bitsandbytes wrappers)
        layer_class_name = layer.__class__.__name__

        if "Linear4bit" in layer_class_name:
            return 4
        if "Linear8bit" in layer_class_name:
            return 8

        # 2. Check custom attributes (used by AutoGPTQ, AWQ, or custom Parameters)
        # Many quantization libraries add a "bits" attribute to the layer or its config
        if hasattr(layer, "bits"):
            return int(getattr(layer, "bits"))

        if hasattr(layer, "quantize_config") and hasattr(layer.quantize_config, "bits"):
            return int(layer.quantize_config.bits)

        # Some custom tensors (like Params4bit in bitsandbytes) store the attribute directly
        if hasattr(tensor, "bits"):
            return int(getattr(tensor, "bits"))

        # 3. Fallback to standard PyTorch dtypes
        if tensor is not None and hasattr(tensor, "dtype"):
            dtype = tensor.dtype
            if dtype in (torch.float64, torch.int64, torch.complex128):
                return 64

            elif dtype in (torch.float32, torch.int32, torch.complex64):
                return 32

            elif dtype in (torch.float16, torch.bfloat16, torch.int16):
                return 16

            elif dtype in (torch.int8, torch.uint8, torch.qint8):
                return 8

            # Support for experimental 4-bit dtypes in PyTorch 2.2+
            elif str(dtype) in ("torch.quint4x2", "torch.int4"):
                return 4

        # Fallback: assume FP32 if the bit-width cannot be inferred
        return 32

    # ------------------------------------------------------------
    # FLOP formulas
    # ------------------------------------------------------------
    # Conv
    def _conv1d_flop(self, conv: nn.Conv1d, x, y):
        batch_size = x.shape[0]
        c_in = conv.in_channels
        c_out = conv.out_channels
        k = conv.kernel_size[0]
        l_out = y.shape[2]
        groups = conv.groups
        flop_per_out = 2 * (c_in // groups) * k
        num_out_elements = batch_size * c_out * l_out
        return flop_per_out * num_out_elements

    def _conv2d_flop(self, conv: nn.Conv2d, x, y):
        batch_size = x.shape[0]
        c_in = conv.in_channels
        c_out = conv.out_channels
        k_h, k_w = conv.kernel_size
        h_out, w_out = y.shape[2], y.shape[3]
        groups = conv.groups
        flop_per_out = 2 * (c_in // groups) * k_h * k_w
        num_out_elements = batch_size * c_out * h_out * w_out
        return flop_per_out * num_out_elements

    def _conv3d_flop(self, conv: nn.Conv3d, x, y):
        batch_size = x.shape[0]
        c_in = conv.in_channels
        c_out = conv.out_channels
        k_d, k_h, k_w = conv.kernel_size
        d_out, h_out, w_out = y.shape[2], y.shape[3], y.shape[4]
        groups = conv.groups
        flop_per_out = 2 * (c_in // groups) * k_d * k_h * k_w
        num_out_elements = batch_size * c_out * d_out * h_out * w_out
        return flop_per_out * num_out_elements

    def _convtranspose1d_flop(self, conv: nn.ConvTranspose1d, x, y):
        batch_size = x.shape[0]
        c_in = conv.in_channels
        c_out = conv.out_channels
        k = conv.kernel_size[0]
        l_out = y.shape[2]
        groups = conv.groups
        flop_per_out = 2 * (c_in // groups) * k
        num_out_elements = batch_size * c_out * l_out
        return flop_per_out * num_out_elements

    def _convtranspose2d_flop(self, conv: nn.ConvTranspose2d, x, y):
        batch_size = x.shape[0]
        c_in = conv.in_channels
        c_out = conv.out_channels
        k_h, k_w = conv.kernel_size
        h_out, w_out = y.shape[2], y.shape[3]
        groups = conv.groups
        flop_per_out = 2 * (c_in // groups) * k_h * k_w
        num_out_elements = batch_size * c_out * h_out * w_out
        return flop_per_out * num_out_elements

    def _convtranspose3d_flop(self, conv: nn.ConvTranspose3d, x, y):
        batch_size = x.shape[0]
        c_in = conv.in_channels
        c_out = conv.out_channels
        k_d, k_h, k_w = conv.kernel_size
        d_out, h_out, w_out = y.shape[2], y.shape[3], y.shape[4]
        groups = conv.groups
        flop_per_out = 2 * (c_in // groups) * k_d * k_h * k_w
        num_out_elements = batch_size * c_out * d_out * h_out * w_out
        return flop_per_out * num_out_elements

    # Linear
    def _linear_flop(self, linear: nn.Linear, x, y):
        batch_size = x.shape[0]
        in_f = linear.in_features
        out_f = linear.out_features
        return batch_size * 2 * in_f * out_f

    # Pooling
    def _pool1d_flop(self, layer, x, y):
        batch_size, c, l_out = y.shape
        if hasattr(layer, "kernel_size"):
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]

        else:
            l_in = x.shape[2]
            k = l_in // l_out if l_out > 0 else 1

        flop_per_out = max(k, 1)
        return batch_size * c * l_out * flop_per_out

    def _pool2d_flop(self, layer, x, y):
        batch_size, c, h_out, w_out = y.shape
        if hasattr(layer, "kernel_size"):
            if isinstance(layer.kernel_size, int):
                k_h = k_w = layer.kernel_size

            else:
                k_h, k_w = layer.kernel_size

        else:
            h_in, w_in = x.shape[2], x.shape[3]
            k_h = max(h_in // h_out, 1)
            k_w = max(w_in // w_out, 1)

        flop_per_out = max(k_h * k_w, 1)
        return batch_size * c * h_out * w_out * flop_per_out

    def _pool3d_flop(self, layer, x, y):
        batch_size, c, d_out, h_out, w_out = y.shape
        if hasattr(layer, "kernel_size"):
            ks = layer.kernel_size
            if isinstance(ks, int):
                k_d = k_h = k_w = ks

            else:
                k_d, k_h, k_w = ks
        else:
            d_in, h_in, w_in = x.shape[2], x.shape[3], x.shape[4]
            k_d = max(d_in // d_out, 1)
            k_h = max(h_in // h_out, 1)
            k_w = max(w_in // w_out, 1)

        flop_per_out = max(k_d * k_h * k_w, 1)
        return batch_size * c * d_out * h_out * w_out * flop_per_out

    # Normalization
    def _batchnorm_flop(self, y):
        return 4 * y.numel()

    def _layernorm_flop(self, y):
        return 4 * y.numel()

    def _groupnorm_flop(self, y):
        return 4 * y.numel()

    def _instancenorm_flop(self, y):
        return 4 * y.numel()

    def _rmsnorm_flop(self, layer, y):
        norm_shape = getattr(layer, "normalized_shape", None)
        if norm_shape is None:
            return 4 * y.numel()

        d = int(norm_shape) if isinstance(norm_shape, int) else int(torch.tensor(norm_shape).prod().item())
        if d <= 0:
            return 0

        vectors = int(y.numel() // d)
        has_bias = getattr(layer, "bias", None) is not None
        ops_per_vec = d + (d - 1) + 1 + 1 + d + d + (d if has_bias else 0)
        return vectors * ops_per_vec

    # Activations
    def _relu_flop(self, y):
        return y.numel()

    def _leakyrelu_flop(self, y):
        return 2 * y.numel()

    def _prelu_flop(self, y):
        return 2 * y.numel()

    def _sigmoid_flop(self, y):
        return 4 * y.numel()

    def _tanh_flop(self, y):
        return 6 * y.numel()

    # Softmax family
    def _softmax_family_flop(self, layer, y):
        if y is None:
            return 0

        if isinstance(layer, nn.Softmax2d):
            if y.dim() != 4:
                return 0

            n, c, h, w = y.shape
            vectors = int(n * h * w)
            k = int(c)

        else:
            dim = getattr(layer, "dim", -1)
            if dim is None or dim < 0:
                dim = -1

            k = int(y.shape[dim])
            vectors = int(y.numel() // k) if k > 0 else 0

        if k <= 0 or vectors <= 0:
            return 0

        softmax_ops = vectors * (k + (k - 1) + k)  # 3k - 1

        if isinstance(layer, nn.Softmin):
            return softmax_ops + vectors * k

        if isinstance(layer, nn.LogSoftmax):
            return softmax_ops + vectors * (k + 1)

        return softmax_ops

    # RNN / LSTM / GRU
    def _rnn_flop(self, layer, x):
        batch_first = getattr(layer, "batch_first", False)
        if batch_first:
            batch_size, seq_len, input_size = x.shape

        else:
            seq_len, batch_size, input_size = x.shape

        hidden_size = layer.hidden_size
        num_layers = layer.num_layers
        num_directions = 2 if layer.bidirectional else 1
        if isinstance(layer, nn.LSTM):
            num_gates = 4

        elif isinstance(layer, nn.GRU):
            num_gates = 3

        else:
            num_gates = 1

        flop_per_timestep = 2 * num_gates * (input_size * hidden_size + hidden_size * hidden_size)
        timesteps = seq_len * num_layers * num_directions
        return batch_size * timesteps * flop_per_timestep

    # Cells
    def _rnncell_flop(self, layer: nn.RNNCell, inputs):
        x = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else None
        hx = inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else None
        if not isinstance(x, torch.Tensor):
            return 0

        b = int(x.shape[0]) if x.dim() >= 2 else 1
        i = int(x.shape[-1])
        h = int(layer.hidden_size)
        fl = 2 * b * h * i
        if isinstance(hx, torch.Tensor):
            fl += 2 * b * h * h

        fl += b * h
        fl += b * h
        return int(fl)

    def _lstmcell_flop(self, layer: nn.LSTMCell, inputs):
        x = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else None
        hx = inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else None
        if not isinstance(x, torch.Tensor):
            return 0

        b = int(x.shape[0]) if x.dim() >= 2 else 1
        i = int(x.shape[-1])
        h = int(layer.hidden_size)
        fl = 4 * (2 * b * h * i)
        if hx is not None:
            fl += 4 * (2 * b * h * h)

        fl += 10 * b * h
        return int(fl)

    def _grucell_flop(self, layer: nn.GRUCell, inputs):
        x = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else None
        hx = inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else None
        if not isinstance(x, torch.Tensor):
            return 0

        b = int(x.shape[0]) if x.dim() >= 2 else 1
        i = int(x.shape[-1])
        h = int(layer.hidden_size)
        fl = 3 * (2 * b * h * i)
        if isinstance(hx, torch.Tensor):
            fl += 3 * (2 * b * h * h)

        fl += 8 * b * h
        return int(fl)

    # MultiheadAttention
    def _mha_flop(self, layer: nn.MultiheadAttention, inputs):
        q = inputs[0]
        k = inputs[1] if len(inputs) > 1 and inputs[1] is not None else q
        l, n, e = q.shape
        s = k.shape[0]
        num_heads = layer.num_heads
        d_k = e // num_heads
        flop_qkv = 3 * 2 * e * e * l * n
        flop_scores = num_heads * 2 * l * s * d_k
        flop_attn_v = num_heads * 2 * l * s * d_k
        flop_out = 2 * e * e * l * n
        return flop_qkv + flop_scores + flop_attn_v + flop_out

    # Embedding
    def _embedding_flop(self, layer: nn.Embedding, x):
        num_indices = x.numel()
        emb_dim = layer.embedding_dim
        return num_indices * emb_dim

    def _embeddingbag_flop(self, layer: nn.EmbeddingBag, x):
        num_indices = x.numel()
        emb_dim = layer.embedding_dim
        return num_indices * emb_dim
