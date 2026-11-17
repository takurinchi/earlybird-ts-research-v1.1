"""Mask serialization helpers."""
import torch


def save_masks(masks, path):
    torch.save({k: v.detach().cpu().to(torch.uint8) for k, v in masks.items()}, path)


def load_masks(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    return {k: v.bool() for k, v in payload.items()}
