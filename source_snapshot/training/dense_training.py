"""Dense training is provided by training.trainer.train_epoch without masks."""
from .trainer import train_epoch


def train_dense_epoch(model, loader, optimizer, criterion, device):
    return train_epoch(model, loader, optimizer, criterion, device, mask_manager=None)
