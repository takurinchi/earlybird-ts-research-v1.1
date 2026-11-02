"""Serialized checkpoint size measurement."""
from pathlib import Path


def model_size_bytes(path):
    return Path(path).stat().st_size
