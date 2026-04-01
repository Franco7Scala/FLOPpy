from __future__ import annotations
from ..base import BaseBackend
from .hooks.universal_flop_counter import UniversalFlopCounter

import torch
import torch.nn as nn


class TorchBackend(BaseBackend):
    """
    Backend for PyTorch models.

    Responsibilities:
    - Handles DataParallel / DDP via .module
    - Counts ONLY model FLOP using TorchDispatchMode (UniversalFlopCounter)
    - Handles quantized layers (BitsAndBytes 4-bit/8-bit) via an escape hatch
    """
    def __init__(self, model: nn.Module, logger=None):
        self._root_model = model
        if isinstance(model, (nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module

        super().__init__(model, logger=logger)
        self._root_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._quant_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._flop_counter: UniversalFlopCounter | None = None
        self.forward_flops_at_last_backward = 0
        self.forward_bops_at_last_backward = 0

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    def start(self):
        """Attaches the root hooks and the quantized escape hatch hooks."""
        # 1. Root hooks to delimit one complete model forward pass
        pre_handle = self._root_model.register_forward_pre_hook(self._on_forward_start)
        post_handle = self._root_model.register_forward_hook(self._on_forward_end)
        self._root_handles.extend([pre_handle, post_handle])
        # 2. Escape hatch hooks for quantized black-box layers
        for name, module in self._root_model.named_modules():
            quantized_classes = [
                "Linear4bit", "Linear8bitLt",  # BitsAndBytes
                "QuantLinear",  # AutoGPTQ
                "WQLinear", "WQLinear_GEMM",  # AutoAWQ
                "HQQLinear",  # HQQ
                "BitLinear"  # 1-bit / 1.58-bit models
            ]
            if module.__class__.__name__ in quantized_classes:
                h_pre = module.register_forward_pre_hook(self._on_quant_pre)
                h_post = module.register_forward_hook(self._on_quant_post)
                self._quant_handles.extend([h_pre, h_post])

        # 3. TRANSPARENT BACKWARD TRACKING (Monkey-Patching)
        # Save the original PyTorch C++ Autograd backward function
        self._original_tensor_backward = torch.Tensor.backward

        def _patched_backward(tensor_obj, *args, **kwargs):
            # Execute the original PyTorch backward pass
            result = self._original_tensor_backward(tensor_obj, *args, **kwargs)

            # We check if the loss tensor has a grad_fn (meaning it's part of a graph)
            # or requires_grad. This is more reliable for Gradient Accumulation.
            if getattr(tensor_obj, "requires_grad", False) or hasattr(tensor_obj, "grad_fn"):

                # 1. INCREMENTAL LOGIC: Get only the forward work done since the last backward
                current_total_f = getattr(self, "total_forward_flop", 0)
                current_total_b = getattr(self, "total_forward_bop", 0)

                step_forward_flops = current_total_f - self.forward_flops_at_last_backward
                step_forward_bops = current_total_b - self.forward_bops_at_last_backward

                # Avoid calculating if there are no new forward FLOPs (prevents double counting)
                if step_forward_flops <= 0:
                    return result

                # 2. PEFT/LoRA RATIO
                trainable_params = sum(p.numel() for p in self._root_model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self._root_model.parameters())
                trainable_ratio = trainable_params / total_params

                # 3. MULTIPLIER (Standard = 2.0, LoRA ~ 1.0)
                backward_multiplier = 1.0 + (1.0 * trainable_ratio)

                flops_to_add = step_forward_flops * backward_multiplier
                bops_to_add = step_forward_bops * backward_multiplier

                # We use the generic "total_backward_flop" which is likely what your report maps to.
                if hasattr(self, "total_backward_flop"):
                    self.total_backward_flop += flops_to_add
                if hasattr(self, "total_backward_bop"):
                    self.total_backward_bop += bops_to_add

                # Also update the specific model-level counters for consistency
                self.total_model_backward_flop = getattr(self, "total_model_backward_flop", 0) + flops_to_add
                self.total_model_backward_bop = getattr(self, "total_model_backward_bop", 0) + bops_to_add

                # 5. SYNC PROGRESS
                self.forward_flops_at_last_backward = current_total_f
                self.forward_bops_at_last_backward = current_total_b

            return result

        # Apply the monkey-patch globally
        torch.Tensor.backward = _patched_backward

    def stop(self):
        """Detaches all hooks and restores monkey-patched methods."""
        for handle in self._root_handles:
            handle.remove()

        self._root_handles.clear()

        for handle in self._quant_handles:
            handle.remove()
        self._quant_handles.clear()

        # Restore the original backward function
        if hasattr(self, "_original_tensor_backward") and self._original_tensor_backward is not None:
            torch.Tensor.backward = self._original_tensor_backward
            self._original_tensor_backward = None

    # ------------------------------------------------------------
    # Root forward hooks (DispatchMode Integration)
    # ------------------------------------------------------------

    def _on_forward_start(self, module, inputs):
        """Initializes and enters the UniversalFlopCounter context manager."""
        self._flop_counter = UniversalFlopCounter()
        self._flop_counter.__enter__()

    def _on_forward_end(self, module, inputs, output):
        """Exits the counter, extracts the metrics, and updates the global states."""
        self._batch_idx += 1
        forward_flop = 0
        forward_bop = 0
        if self._flop_counter is not None:
            # Close the context manager to stop intercepting operations
            self._flop_counter.__exit__(None, None, None)
            # Extract the calculated totals
            forward_flop = int(getattr(self._flop_counter, "flops", 0))
            forward_bop = int(getattr(self._flop_counter, "bops", 0))

        # Update batch metrics
        self._last_batch_flop = forward_flop
        self._last_batch_bop = forward_bop
        # Update global accumulated metrics (inherited from BaseBackend)
        self.total_forward_flop += forward_flop
        self.total_forward_bop += forward_bop

    # ------------------------------------------------------------
    # Quantized "Escape Hatch" hooks
    # ------------------------------------------------------------

    def _on_quant_pre(self, module, inputs):
        """Pauses the dispatcher right before BitsAndBytes performs its custom computations."""
        if self._flop_counter is not None:
            self._flop_counter.paused = True

    def _on_quant_post(self, module, inputs, output):
        """Calculates FLOPs/BOPs manually, including dequantization overhead, and resumes the dispatcher."""
        if self._flop_counter is not None:
            x = inputs[0]
            if isinstance(x, torch.Tensor):
                in_f = module.in_features
                out_f = module.out_features
                # Dynamic calculation for any batch or sequence size
                batch_elements = x.numel() // in_f
                # 1. Pure Matrix Multiplication FLOPs (Algorithmic complexity)
                matmul_flops = 2 * batch_elements * in_f * out_f
                # 2. Dequantization Overhead (Hardware Just-In-Time unpacking)
                # Every weight in the matrix must be unpacked and scaled (approx. 1 FLOP per weight)
                num_weights = in_f * out_f
                dequant_flops = num_weights
                total_flops = matmul_flops + dequant_flops
                # 3. Calculate Bit Operations (BOPs)
                # The bit-width is 4 for Linear4bit and 8 for Linear8bitLt
                if hasattr(module, "bits"):
                    bit_width = getattr(module, "bits")

                elif module.__class__.__name__ == "Linear8bitLt":
                    bit_width = 8

                elif module.__class__.__name__ == "BitLinear":
                    bit_width = 1

                else:
                    bit_width = 4

                self._flop_counter.flops += total_flops
                # BOPs heuristic:
                # The MatMul represents the algorithmic load scaled down by the reduced bit-width,
                # while the dequantization overhead itself is processed using standard 16-bit logic.
                self._flop_counter.bops += (matmul_flops * bit_width) + (dequant_flops * 16)

            # Unpause the dispatcher so it resumes listening to standard PyTorch operations
            self._flop_counter.paused = False
