"""Resolve requested compute device, rejecting unavailable CUDA explicitly."""
import torch


def resolve_device(requested=None):
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment")
    return device
