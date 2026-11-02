"""Checkpoint helpers; saves enough metadata to reproduce a run."""
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer=None, epoch=None, metadata=None):
    payload = {"model": model.state_dict(), "epoch": epoch,
               "metadata": metadata or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
