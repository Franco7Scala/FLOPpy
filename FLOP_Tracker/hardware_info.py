from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import os
import platform
import sys

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None


@dataclass
class HardwareInfo:
    os: str
    os_version: str
    machine: str
    processor: str
    python_version: str
    cpu_cores_logical: Optional[int]
    cpu_cores_physical: Optional[int]
    ram_total_gb: Optional[float]
    cuda_available: Optional[bool]
    gpu_name: Optional[str]
    gpu_count: Optional[int]
    torch_version: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _bytes_to_gb(x: int) -> float:
    return round(x / (1024 ** 3), 2)


def get_hardware_info() -> HardwareInfo:
    os_name = platform.system()
    os_ver = platform.version()
    machine = platform.machine()
    processor = platform.processor()
    pyver = sys.version.split()[0]

    cpu_logical = None
    cpu_physical = None
    ram_gb = None

    if psutil is not None:
        try:
            cpu_logical = psutil.cpu_count(logical=True)
            cpu_physical = psutil.cpu_count(logical=False)
            ram_gb = _bytes_to_gb(psutil.virtual_memory().total)
        except Exception:
            pass

    cuda_avail = None
    gpu_name = None
    gpu_count = None
    torch_ver = None

    if torch is not None:
        try:
            torch_ver = getattr(torch, "__version__", None)
            cuda_avail = bool(torch.cuda.is_available())
            if cuda_avail:
                gpu_count = int(torch.cuda.device_count())
                gpu_name = torch.cuda.get_device_name(0) if gpu_count and gpu_count > 0 else None
        except Exception:
            pass

    return HardwareInfo(
        os=os_name,
        os_version=os_ver,
        machine=machine,
        processor=processor,
        python_version=pyver,
        cpu_cores_logical=cpu_logical,
        cpu_cores_physical=cpu_physical,
        ram_total_gb=ram_gb,
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        torch_version=torch_ver,
    )


def format_hardware_info(hw: HardwareInfo) -> str:
    parts = [
        f"OS: {hw.os} ({hw.machine})",
        f"Python: {hw.python_version}",
    ]
    if hw.torch_version:
        parts.append(f"PyTorch: {hw.torch_version}")

    if hw.cpu_cores_logical or hw.cpu_cores_physical:
        parts.append(
            f"CPU cores: logical={hw.cpu_cores_logical}, physical={hw.cpu_cores_physical}"
        )
    if hw.ram_total_gb is not None:
        parts.append(f"RAM: {hw.ram_total_gb} GB")

    if hw.cuda_available is True:
        parts.append(f"CUDA: yes (gpus={hw.gpu_count}, name={hw.gpu_name})")
    elif hw.cuda_available is False:
        parts.append("CUDA: no")

    return " | ".join(parts)
