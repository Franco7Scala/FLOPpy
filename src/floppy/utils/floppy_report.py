from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from floppy.utils.system_info import SystemInfo
from floppy.utils.wandb_configuration import WandbConfiguration


@dataclass
class FLOPpyReport:
    """
        A comprehensive report containing the aggregated computational workload and hardware statistics of a monitored machine learning run.
    """

    # The custom name assigned to this tracking session.
    run_name: Optional[str]
    # The structural class name of the monitored model (e.g., 'ResNet', 'RandomForest')
    model_architecture: Optional[str]
    # The hardware device where the model is allocated (e.g., 'cpu', 'cuda:0').
    model_device: Optional[str]
    # The name of the loss function used during training.
    loss_type: Optional[str]
    # The name of the optimizer used for weight updates.
    optimizer_type: Optional[str]
    # The specific FLOPpy backend utilized (e.g., 'pytorch', 'sklearn').
    backend: str
    # The total Floating Point Operations consumed by the model's structural layers.
    model_flop: int
    # The computational overhead (in FLOPs) introduced by the optimizer step.
    optimizer_flop: int
    # The FLOPs consumed during the loss function evaluation.
    loss_forward_flop: int
    # The FLOPs consumed during the loss gradient computation.
    loss_backward_flop: int
    # The operations workload for input preparation (e.g., tokenization steps).
    preproc_ops: int
    # The total aggregated FLOPs across the entire tracked pipeline.
    overall_flop: int
    # The file system path where the CSV report is saved, if applicable.
    export_path: Optional[str]
    # The Weights & Biases configuration used for real-time logging.
    wandb_config: Optional[WandbConfiguration]
    # A snapshot of the execution environment's specifications.
    system: SystemInfo
    
    def __str__(self):
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
        # System Info
        if self.system is not None:
            h = self.system
            result += "System Environment:\n"
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
                frameworks.append(f"PyTorch {h.torch_version}")

            if h.sklearn_version:
                frameworks.append(f"Scikit-learn {h.sklearn_version}")

            if frameworks:
                result += f"  - Libs     : {' | '.join(frameworks)}\n"

        # Model and Device details
        result += "Modules tracked details:\n"
        result += f"  - Device   : {self.model_device}\n"
        result += f"  - Model    : {self.model_architecture}\n"
        if self.loss_type is not None:
            result += f"  - Loss     : {self.loss_type}\n"

        if self.optimizer_type is not None:
            result += f"  - Optimizer: {self.optimizer_type}\n"

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
        if self.export_path or self.wandb_config:
            result += "Tracking & Integrations:\n"
            if self.export_path:
                result += f"  - Export Path: {self.export_path}\n"

            if self.wandb_config:
                result += f"  - W&B Project: {self.wandb_config.project_name}\n"
                result += f"  - W&B group: {self.wandb_config.group_name}\n"

            result += "=" * 70 + "\n"

        return result
