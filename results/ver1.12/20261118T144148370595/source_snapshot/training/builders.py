"""Construct the CIFAR-paper candidate optimizer and epoch-step schedule."""
import torch


def build_optimizer(model, config):
    name = str(config.get("optimizer", "SGD")).lower()
    if name != "sgd":
        raise ValueError(f"Unsupported optimizer candidate: {name}")
    return torch.optim.SGD(model.parameters(),
        lr=float(config["learning_rate"]), momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]))


def build_scheduler(optimizer, config):
    schedule = config["lr_schedule"]
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(schedule["milestones"]), gamma=float(schedule["gamma"]))
