"""Sparse epoch adapter with persistent masks."""
from .trainer import train_epoch
from ..pruning.mask_manager import MaskManager


def train_sparse_epoch(model, loader, optimizer, criterion, device, mask_manager):
    if not isinstance(mask_manager, MaskManager):
        raise TypeError("mask_manager must be a MaskManager")
    return train_epoch(model, loader, optimizer, criterion, device, mask_manager)
