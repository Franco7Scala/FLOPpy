from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import platform
import sys
import psutil
import torch
import sklearn
import subprocess


@dataclass
class SystemInfo:
    """
        A snapshot of the hardware and software environment where the model is executed.

        Attributes:
            os (str): The name of the operating system (e.g., 'Linux', 'Windows').
            os_version (str): The specific release version of the operating system.
            machine (str): The hardware architecture of the machine (e.g., 'x86_64', 'AMD64').
            processor (str): The processor family or detailed name provided by the system.
            python_version (str): The version of Python currently running the tracker.
            cpu_name (str, optional): The specific model name of the CPU (e.g., 'Intel Core i9').
            cpu_cores_logical (int, optional): The number of logical CPU cores (including hyperthreading).
            cpu_cores_physical (int, optional): The number of physical CPU cores.
            ram_total_gb (float, optional): The total amount of system RAM available, expressed in Gigabytes.
            cuda_available (bool, optional): Indicates whether a CUDA-enabled GPU is accessible to PyTorch.
            gpu_name (str, optional): The model name of the primary GPU detected (e.g., 'NVIDIA RTX 4090').
            gpu_count (int, optional): The total number of GPUs available on the system.
            torch_version (str, optional): The installed version of the PyTorch library.
            sklearn_version (str, optional): The installed version of the Scikit-learn library.
    """
    os: str
    os_version: str
    machine: str
    processor: str
    python_version: str
    cpu_name: Optional[str]
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


def _get_cpu_name() -> Optional[str]:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()

        elif system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").strip()

        elif system == "Windows":
            output = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode("utf-8").strip()

            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if len(lines) > 1:
                return lines[1]

    except Exception:
        pass

    # Fallback if the OS-specific commands fail
    return platform.processor() or "Unknown CPU"

def get_system_info() -> SystemInfo:
    os_name = platform.system()
    os_ver = platform.version()
    machine = platform.machine()
    processor = platform.processor()
    pyver = sys.version.split()[0]
    cpu_name = _get_cpu_name()
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

    return SystemInfo(
        os=os_name,
        os_version=os_ver,
        machine=machine,
        processor=processor,
        python_version=pyver,
        cpu_name=cpu_name,
        cpu_cores_logical=cpu_logical,
        cpu_cores_physical=cpu_physical,
        ram_total_gb=ram_gb,
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        torch_version=torch_ver,
        sklearn_version=sklearn_ver
    )
