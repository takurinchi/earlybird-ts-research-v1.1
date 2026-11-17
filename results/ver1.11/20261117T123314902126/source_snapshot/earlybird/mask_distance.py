"""Mask distance utilities. Verify this default against the inherited EB code."""
from collections.abc import Mapping


def _flatten_masks(masks):
    if isinstance(masks, Mapping):
        values = [masks[k] for k in sorted(masks)]
    elif isinstance(masks, (list, tuple)):
        values = list(masks)
    else:
        values = [masks]
    if not values:
        raise ValueError("At least one mask tensor is required")
    try:
        import torch
        values = [v.detach().reshape(-1).to(dtype=torch.bool, device="cpu") for v in values]
        return torch.cat(values)
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for tensor masks") from exc


def mask_distance(previous, current) -> float:
    """Return binary Hamming disagreement / element count, in [0, 1]."""
    if isinstance(previous, Mapping) and isinstance(current, Mapping):
        if set(previous) != set(current):
            raise ValueError("Corresponding mask mappings must have the same keys")
        for key in previous:
            if tuple(previous[key].shape) != tuple(current[key].shape):
                raise ValueError(f"Corresponding mask shapes differ for {key}")
    elif isinstance(previous, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(previous) != len(current):
            raise ValueError("Corresponding mask sequences must have equal lengths")
        if any(tuple(a.shape) != tuple(b.shape) for a, b in zip(previous, current)):
            raise ValueError("Corresponding mask shapes differ")
    a, b = _flatten_masks(previous), _flatten_masks(current)
    if a.numel() != b.numel():
        raise ValueError("Masks must have the same total number of elements")
    return float((a != b).float().mean().item())
