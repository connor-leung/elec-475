"""
ELEC 475: Lab 3 - Device Selection Utility
Picks the fastest available backend: CUDA (NVIDIA), MPS (Apple Silicon) or CPU.
"""
import subprocess
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        except OSError:
            chip = ""
        return f"{chip or 'Apple Silicon'} GPU"
    return "CPU"


def synchronize(device: torch.device):
    """Waits for queued GPU work to finish so wall-clock timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
