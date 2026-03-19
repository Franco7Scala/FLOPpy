from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from floppy.utils.hardware_info import HardwareInfo


@dataclass
class FLOPpyReport:
    run_name: Optional[str]
    model_architecture: Optional[str]
    model_device: Optional[str]
    #loss_type: Optional[str]
    #optimizer_type: Optional[str]
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
    
    def __str__(self):
        return ""
        def format_flops(flops: int) -> str:
            if flops == 0:
                return "0 FLOPs"

            units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
            unit_idx = 0
            float_flops = float(flops)
            while float_flops >= 1000.0 and unit_idx < len(units) - 1:
                float_flops /= 1000.0
                unit_idx += 1

            return f"{float_flops:.2f} {units[unit_idx]}"

        result = ""
        run_label = f"'{self.run_name}'" if self.run_name else ""
        result += "=" * 70 + "\n"
        result += f" FLOPpyTracker Summary{run_label}\n"
        result += "=" * 70 + "\n"
        # Hardware Info
        if self.hardware is not None:
            h = self.hardware
            result += "Hardware Environment:\n"
            # System & RAM
            ram_str = f"{h.ram_total_gb:.0f} GB RAM" if h.ram_total_gb else "Unknown RAM"
            result += f"  - System   : {h.os} ({h.machine}) | {ram_str}\n"
            # CPU Details
            c_name = getattr(h, "cpu_name", None) or h.processor or "Unknown CPU"
            cores_str = f"{h.cpu_cores_physical} Physical Cores" if h.cpu_cores_physical else "Unknown Cores"
            result += f"  - CPU      : {c_name} | {cores_str}\n"
            # GPU Details
            if h.cuda_available:
                g_count = h.gpu_count or 1
                g_name = h.gpu_name or "Unknown GPU"
                result += f"  - GPU      : {g_count}x {g_name}\n"
            else:
                result += "  - GPU      : None (CPU Only)\n"

            # Software
            result += f"  - Python   : {h.python_version}\n"

            frameworks = []
            if h.torch_version:
                frameworks.append(f"PyTorch {h.torch_version}\n")

            if h.sklearn_version:
                frameworks.append(f"Scikit-learn {h.sklearn_version}\n")

            if frameworks:
                result += f"  - Libs     : {' | '.join(frameworks)}\n"

        # Model and Device details
        result += "Modules tracked details:"
        result += f"  - Device   : {self.model_device}\n"
        result += f"  - Model    : {self.model_architecture}\n"
        #result += f"  - Loss     : {self.loss_type}\n")
        #result += f"  - Optimizer: {self.optimizer_type}\n")

        # Computational Workload Breakdown
        result += "Computational Workload Breakdown:\n"
        result += f"  - Model (Forward)         : {format_flops(self.model_flop):>15}\n"
        if self.loss_forward_flop > 0:
            result += f"  - Loss (Forward)          : {format_flops(self.loss_forward_flop):>15}\n"

        if self.loss_backward_flop > 0:
            result += f"  - Loss (Backward)         : {format_flops(self.loss_backward_flop):>15}\n"

        if self.optimizer_flop > 0:
            result += f"  - Optimizer (Update)      : {format_flops(self.optimizer_flop):>15}\n"

        if self.preproc_ops > 0:
            result += f"  - Preprocessing/Tokenizer : {str(self.preproc_ops) + ' Ops':>15}\n"

        result += "-" * 70 + "\n"
        # Totals
        result += f"OVERALL TOTAL FLOPs         : {format_flops(self.overall_flop):>15}\n"
        result += "=" * 70 + "\n"
        # Integrations
        if self.export_path or (self.use_wandb and self.wandb_project):
            result += "Tracking & Integrations:\n"
            if self.export_path:
                result += f"  - Export Path: {self.export_path}\n"

            if self.use_wandb and self.wandb_project:
                result += f"  - W&B Project: {self.wandb_project}\n"

            result += "=" * 70 + "\n"

        return result
