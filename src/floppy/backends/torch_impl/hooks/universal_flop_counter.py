import torch
from torch.utils._python_dispatch import TorchDispatchMode

try:
    from ._flop_counter_core import compute_operation
except Exception:
    compute_operation = None


class UniversalFlopCounter(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.flops = 0
        self.bops = 0
        self.paused = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if self.paused:
            return func(*args, **kwargs)

        out = func(*args, **kwargs)

        if compute_operation is not None:
            delta_flops, delta_bops = compute_operation(func, args, out)
            self.flops += int(delta_flops)
            self.bops += int(delta_bops)

        return out