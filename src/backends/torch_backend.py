from __future__ import annotations
import torch
import torch.nn as nn

from .base import BaseBackend


class TorchBackend(BaseBackend):
    """
    Backend per modelli PyTorch.
    - Gestisce DataParallel / DDP usando .module
    - Conta i FLOP di:
        * Conv1d / Conv2d / Conv3d
        * ConvTranspose1d / 2d / 3d
        * Linear
        * Pooling (Max/Avg/Adaptive)
        * Normalization (BatchNorm, LayerNorm, GroupNorm, InstanceNorm)
        * RMSNorm
        * Activations: ReLU, LeakyReLU, PReLU, Sigmoid, Tanh
        * Softmax family: Softmax, Softmin, Softmax2d, LogSoftmax
        * RNN / LSTM / GRU
        * RNNCell / LSTMCell / GRUCell
        * MultiheadAttention
        * Embedding / EmbeddingBag
        * Transformer 
        * DataParallel
        
        Per i container / wrapper (Transformer e DataParallel) restituisce FLOP = 0, conteggiando i FLOP reali dai sotto-moduli già hookati.
    - Logga per batch / epoch / wandb tramite logger (se presente)
    """

    def __init__(self, model: nn.Module, logger=None):
        # DataParallel / DDP
        if isinstance(model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module

        super().__init__(model, logger=logger)
        self._layer_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._root_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current_batch_flop: int = 0

    # ---------------- START / STOP ---------------- #

    def start(self):
        # Hook sui layer da tracciare
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    # --- Convoluzioni / Linear ---
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                    nn.Linear,

                    # --- Pooling ---
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

                    # --- Normalization ---
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.LayerNorm,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,

                    # --- Activations ---
                    nn.ReLU,
                    nn.LeakyReLU,
                    nn.PReLU,
                    nn.Sigmoid,
                    nn.Tanh,

                    # --- Softmax family ---
                    nn.Softmax,
                    nn.Softmin,
                    nn.Softmax2d,
                    nn.LogSoftmax,

                    # --- RNN family ---
                    nn.RNN,
                    nn.LSTM,
                    nn.GRU,
                    nn.RNNCell,
                    nn.LSTMCell,
                    nn.GRUCell,

                    # --- Attention / Embeddings ---
                    nn.MultiheadAttention,
                    nn.Embedding,
                    nn.EmbeddingBag,

                    # --- Transformer high-level containers (FLOP = 0 qui) ---
                    nn.Transformer,
                    nn.TransformerEncoder,
                    nn.TransformerDecoder,
                    nn.TransformerEncoderLayer,
                    nn.TransformerDecoderLayer,

                    # --- DataParallel wrapper (FLOP = 0 qui) ---
                    nn.DataParallel,
                ),
            ):
                h = module.register_forward_hook(self._layer_hook)
                self._layer_handles.append(h)

        # RMSNorm
        if hasattr(nn, "RMSNorm"):
            for module in self.model.modules():
                if isinstance(module, nn.RMSNorm):
                    h = module.register_forward_hook(self._layer_hook)
                    self._layer_handles.append(h)

        # Hook sul modello root per identificare inizio/fine batch
        pre_h = self.model.register_forward_pre_hook(self._on_batch_start)
        post_h = self.model.register_forward_hook(self._on_batch_end)
        self._root_handles.extend([pre_h, post_h])

    def stop(self):
        for h in self._layer_handles:
            h.remove()
        for h in self._root_handles:
            h.remove()
        self._layer_handles.clear()
        self._root_handles.clear()

    def add_extra_flop(self, flop: int) -> None:
    """
    Aggiunge FLOP esterni (es. loss) al batch corrente.
    Verranno sommati al totale a fine batch (_on_batch_end).
    """
    if flop is None:
        return
    fl = int(flop)
    if fl <= 0:
        return
    self._current_batch_flop += fl

    # ---------------- HOOK DI BATCH ---------------- #

    def _on_batch_start(self, module, input):
        self._current_batch_flop = 0

    def _on_batch_end(self, module, input, output):
        batch_flop = self._current_batch_flop
        self._last_batch_flop = batch_flop
        self.total_flop += batch_flop
        self._batch_idx += 1

        if self.logger is not None and hasattr(self.logger, "log_batch"):
            self.logger.log_batch(
                step=self._batch_idx,
                flop=batch_flop,
                cumulative_flop=self.total_flop,
                epoch=self._epoch_idx,
            )

    # ---------------- HOOK DEI LAYER ---------------- #

    def _layer_hook(self, layer, input, output):
        # alcuni layer (es. MHA) hanno input/output non banali
        x = input[0] if isinstance(input, (tuple, list)) and len(input) > 0 else None
        y = output

        # se output è tuple/list, si prende il primo tensore
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
            flop = self._batchnorm_flop(layer, x, y)
        elif isinstance(layer, nn.LayerNorm):
            flop = self._layernorm_flop(layer, x, y)
        elif isinstance(layer, nn.GroupNorm):
            flop = self._groupnorm_flop(layer, x, y)
        elif isinstance(layer, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            flop = self._instancenorm_flop(layer, x, y)
        elif hasattr(nn, "RMSNorm") and isinstance(layer, nn.RMSNorm):
            flop = self._rmsnorm_flop(layer, x, y)

        # --- ACTIVATIONS --- #
        elif isinstance(layer, nn.ReLU):
            flop = self._relu_flop(x, y)
        elif isinstance(layer, nn.LeakyReLU):
            flop = self._leakyrelu_flop(x, y)
        elif isinstance(layer, nn.PReLU):
            flop = self._prelu_flop(x, y)
        elif isinstance(layer, nn.Sigmoid):
            flop = self._sigmoid_flop(x, y)
        elif isinstance(layer, nn.Tanh):
            flop = self._tanh_flop(x, y)

        # --- SOFTMAX FAMILY --- #
        elif isinstance(layer, (nn.Softmax, nn.Softmin, nn.Softmax2d, nn.LogSoftmax)):
            flop = self._softmax_family_flop(layer, x, y)

        # --- RNN / LSTM / GRU --- #
        elif isinstance(layer, (nn.RNN, nn.LSTM, nn.GRU)):
            flop = self._rnn_flop(layer, x, output)

        # --- RNN CELLS --- #
        elif isinstance(layer, nn.RNNCell):
            flop = self._rnncell_flop(layer, input, y)
        elif isinstance(layer, nn.LSTMCell):
            flop = self._lstmcell_flop(layer, input, output)
        elif isinstance(layer, nn.GRUCell):
            flop = self._grucell_flop(layer, input, y)

        # --- MULTIHEAD ATTENTION --- #
        elif isinstance(layer, nn.MultiheadAttention):
            flop = self._mha_flop(layer, input, output)

        # --- EMBEDDING --- #
        elif isinstance(layer, nn.Embedding):
            flop = self._embedding_flop(layer, x, y)
        elif isinstance(layer, nn.EmbeddingBag):
            flop = self._embeddingbag_flop(layer, x, y)

        # --- TRANSFORMER CONTAINERS / DATAPARALLEL WRAPPER --- #
        # sono container, si contano i sotto-moduli già hookati
        elif isinstance(layer, (nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder,
                                nn.TransformerEncoderLayer, nn.TransformerDecoderLayer, nn.DataParallel)):
            flop = 0

        self._current_batch_flop += int(flop)

    # ---------------- FORMULE FLOP ---------------- #
    # Conv

    def _conv1d_flop(self, conv: nn.Conv1d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K = conv.kernel_size[0]
        L_out = y.shape[2]
        groups = conv.groups
        flop_per_out = 2 * (C_in // groups) * K
        num_out_elements = batch_size * C_out * L_out
        return flop_per_out * num_out_elements

    def _conv2d_flop(self, conv: nn.Conv2d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K_h, K_w = conv.kernel_size
        H_out, W_out = y.shape[2], y.shape[3]
        groups = conv.groups
        flop_per_out = 2 * (C_in // groups) * K_h * K_w
        num_out_elements = batch_size * C_out * H_out * W_out
        return flop_per_out * num_out_elements

    def _conv3d_flop(self, conv: nn.Conv3d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K_d, K_h, K_w = conv.kernel_size
        D_out, H_out, W_out = y.shape[2], y.shape[3], y.shape[4]
        groups = conv.groups
        flop_per_out = 2 * (C_in // groups) * K_d * K_h * K_w
        num_out_elements = batch_size * C_out * D_out * H_out * W_out
        return flop_per_out * num_out_elements

    def _convtranspose1d_flop(self, conv: nn.ConvTranspose1d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K = conv.kernel_size[0]
        L_out = y.shape[2]
        groups = conv.groups
        flop_per_out = 2 * (C_in // groups) * K
        num_out_elements = batch_size * C_out * L_out
        return flop_per_out * num_out_elements

    def _convtranspose2d_flop(self, conv: nn.ConvTranspose2d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K_h, K_w = conv.kernel_size
        H_out, W_out = y.shape[2], y.shape[3]
        groups = conv.groups
        flop_per_out = 2 * (C_in // groups) * K_h * K_w
        num_out_elements = batch_size * C_out * H_out * W_out
        return flop_per_out * num_out_elements

    def _convtranspose3d_flop(self, conv: nn.ConvTranspose3d, x, y):
        batch_size = x.shape[0]
        C_in = conv.in_channels
        C_out = conv.out_channels
        K_d, K_h, K_w = conv.kernel_size
        D_out, H_out, W_out = y.shape[2], y.shape[3], y.shape[4]
        groups = conv.groups
        flop_per_out = 2 * (C_in // groups) * K_d * K_h * K_w
        num_out_elements = batch_size * C_out * D_out * H_out * W_out
        return flop_per_out * num_out_elements

    # Linear

    def _linear_flop(self, linear: nn.Linear, x, y):
        batch_size = x.shape[0]
        in_f = linear.in_features
        out_f = linear.out_features
        return batch_size * 2 * in_f * out_f

    # Pooling

    def _pool1d_flop(self, layer, x, y):
        batch_size, C, L_out = y.shape
        if hasattr(layer, "kernel_size"):
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
        else:
            L_in = x.shape[2]
            k = L_in // L_out if L_out > 0 else 1
        k_eff = max(k, 1)
        flop_per_out = k_eff
        return batch_size * C * L_out * flop_per_out

    def _pool2d_flop(self, layer, x, y):
        batch_size, C, H_out, W_out = y.shape
        if hasattr(layer, "kernel_size"):
            if isinstance(layer.kernel_size, int):
                K_h = K_w = layer.kernel_size
            else:
                K_h, K_w = layer.kernel_size
        else:
            H_in, W_in = x.shape[2], x.shape[3]
            K_h = max(H_in // H_out, 1)
            K_w = max(W_in // W_out, 1)
        k_eff = max(K_h * K_w, 1)
        flop_per_out = k_eff
        return batch_size * C * H_out * W_out * flop_per_out

    def _pool3d_flop(self, layer, x, y):
        batch_size, C, D_out, H_out, W_out = y.shape
        if hasattr(layer, "kernel_size"):
            ks = layer.kernel_size
            if isinstance(ks, int):
                K_d = K_h = K_w = ks
            else:
                K_d, K_h, K_w = ks
        else:
            D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
            K_d = max(D_in // D_out, 1)
            K_h = max(H_in // H_out, 1)
            K_w = max(W_in // W_out, 1)
        k_eff = max(K_d * K_h * K_w, 1)
        flop_per_out = k_eff
        return batch_size * C * D_out * H_out * W_out * flop_per_out

    # Normalization (stima: ~4 FLOP per elemento)
    def _batchnorm_flop(self, layer, x, y):
        return 4 * y.numel()

    def _layernorm_flop(self, layer, x, y):
        return 4 * y.numel()

    def _groupnorm_flop(self, layer, x, y):
        return 4 * y.numel()

    def _instancenorm_flop(self, layer, x, y):
        return 4 * y.numel()

    # RMSNorm (stima dedicata)
    def _rmsnorm_flop(self, layer, x, y):
        # normalized_shape può essere int o tuple
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
    def _relu_flop(self, x, y):
        return y.numel()

    def _leakyrelu_flop(self, x, y):
        return 2 * y.numel()

    def _prelu_flop(self, x, y):
        return 2 * y.numel()

    def _sigmoid_flop(self, x, y):
        return 4 * y.numel()

    def _tanh_flop(self, x, y):
        return 6 * y.numel()

    # Softmax family
    def _softmax_family_flop(self, layer, x, y):
        if y is None:
            return 0

        # Softmax2d: normalize over C in (N,C,H,W)
        if isinstance(layer, nn.Softmax2d):
            if y.dim() != 4:
                return 0
            N, C, H, W = y.shape
            vectors = int(N * H * W)
            k = int(C)
        else:
            dim = getattr(layer, "dim", -1)
            if dim is None or dim < 0:
                dim = -1
            k = int(y.shape[dim])
            vectors = int(y.numel() // k) if k > 0 else 0

        if k <= 0 or vectors <= 0:
            return 0

        # softmax per vettore:
        # exp k + sum(k-1) + div k
        softmax_ops = vectors * (k + (k - 1) + k)  # (3k - 1)

        if isinstance(layer, nn.Softmin):
            # softmin(x)=softmax(-x): aggiungo k negazioni
            return softmax_ops + vectors * k

        if isinstance(layer, nn.LogSoftmax):
            # logsoftmax: stima = softmax + (sub k + log 1)
            return softmax_ops + vectors * (k + 1)

        return softmax_ops

    # RNN / LSTM / GRU (stima classica per gate)
    def _rnn_flop(self, layer, x, y):
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

    # RNNCell / LSTMCell / GRUCell
    def _rnncell_flop(self, layer: nn.RNNCell, inputs, y):
        x = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else None
        hx = inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else None
        if not isinstance(x, torch.Tensor):
            return 0
        B = int(x.shape[0]) if x.dim() >= 2 else 1
        I = int(x.shape[-1])
        H = int(layer.hidden_size)
        fl = 2 * B * H * I
        if isinstance(hx, torch.Tensor):
            fl += 2 * B * H * H
        fl += B * H  # bias
        fl += B * H  # nonlinearity
        return int(fl)

    def _lstmcell_flop(self, layer: nn.LSTMCell, inputs, output):
        x = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else None
        hx = inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else None
        if not isinstance(x, torch.Tensor):
            return 0
        B = int(x.shape[0]) if x.dim() >= 2 else 1
        I = int(x.shape[-1])
        H = int(layer.hidden_size)
        fl = 4 * (2 * B * H * I)
        # hx può essere (h,c) oppure Tensor;
        if hx is not None:
            fl += 4 * (2 * B * H * H)
        fl += 10 * B * H  # attivazioni + combinazioni 
        return int(fl)

    def _grucell_flop(self, layer: nn.GRUCell, inputs, y):
        x = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) > 0 else None
        hx = inputs[1] if isinstance(inputs, (tuple, list)) and len(inputs) > 1 else None
        if not isinstance(x, torch.Tensor):
            return 0
        B = int(x.shape[0]) if x.dim() >= 2 else 1
        I = int(x.shape[-1])
        H = int(layer.hidden_size)
        fl = 3 * (2 * B * H * I)
        if isinstance(hx, torch.Tensor):
            fl += 3 * (2 * B * H * H)
        fl += 8 * B * H  # attivazioni + combinazioni 
        return int(fl)

    # MultiheadAttention (stima)
    def _mha_flop(self, layer: nn.MultiheadAttention, input, output):
        q = input[0]
        k = input[1] if len(input) > 1 and input[1] is not None else q
        v = input[2] if len(input) > 2 and input[2] is not None else q

        L, N, E = q.shape
        S = k.shape[0]

        num_heads = layer.num_heads
        d_k = E // num_heads

        flop_qkv = 3 * 2 * E * E * L * N
        flop_scores = num_heads * 2 * L * S * d_k
        flop_attn_v = num_heads * 2 * L * S * d_k
        flop_out = 2 * E * E * L * N

        return flop_qkv + flop_scores + flop_attn_v + flop_out

    # Embedding (lookup: 1 FLOP per valore estratto)
    def _embedding_flop(self, layer: nn.Embedding, x, y):
        num_indices = x.numel()
        emb_dim = layer.embedding_dim
        return num_indices * emb_dim

    def _embeddingbag_flop(self, layer: nn.EmbeddingBag, x, y):
        num_indices = x.numel()
        emb_dim = layer.embedding_dim
        return num_indices * emb_dim
