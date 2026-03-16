from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional


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
            if h is not None:
                h.remove()
