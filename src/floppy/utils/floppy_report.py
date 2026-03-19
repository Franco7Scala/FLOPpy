from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from floppy.utils.hardware_info import HardwareInfo


@dataclass
class FLOPpyReport:
    run_name: Optional[str]
    model_architecture: Optional[str]
    model_device: Optional[str]
    backend: str
    model_flop: int
    optimizer_flop: int
    loss_forward_flop: int
    loss_backward_flop: int
    preproc_ops: int
    overall_flop: int
    export_path: Optional[str]
    use_wandb: bool
    wandb_project: Optional[str]
    hardware: HardwareInfo
