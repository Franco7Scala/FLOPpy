from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import platform
import sys
import psutil
import torch
import sklearn


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
    sklearn_version: Optional[str]


def _bytes_to_gb(x: int) -> float:
    return round(x / (1024 ** 3), 2)


def get_hardware_info() -> HardwareInfo:
    os_name = platform.system()
    os_ver = platform.version()
    machine = platform.machine()
    processor = platform.processor()
    pyver = sys.version.split()[0]
    cpu_logical = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    ram_gb = _bytes_to_gb(psutil.virtual_memory().total)
    sklearn_ver = getattr(sklearn, "__version__", None)
    torch_ver = getattr(torch, "__version__", None)
    cuda_avail = bool(torch.cuda.is_available())

    if cuda_avail:
        gpu_count = int(torch.cuda.device_count())
        gpu_name = torch.cuda.get_device_name(0) if gpu_count and gpu_count > 0 else None

    else:
        gpu_count = None
        gpu_name = None

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
        sklearn_version=sklearn_ver
    )
