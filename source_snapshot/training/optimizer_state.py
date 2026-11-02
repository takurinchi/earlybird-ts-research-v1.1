"""Transfer optimizer and scheduler state when a VGG model is compacted."""
from copy import deepcopy

import torch


def _slice_state_tensor(value, old_shape, new_shape, selectors, name):
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(old_shape):
        return value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
    result = value
    for dim, index in selectors:
        result = result.index_select(int(dim), index.to(device=result.device, dtype=torch.long))
    if tuple(result.shape) != tuple(new_shape):
        raise ValueError(
            f"Optimizer state for {name} has shape {tuple(result.shape)}; "
            f"expected compact shape {tuple(new_shape)}")
    return result.detach().clone()


def migrate_optimizer_state(old_optimizer, new_optimizer, old_model, new_model,
                           parameter_slices):
    """Move optimizer state by parameter name, slicing momentum with weights.

    Supports SGD momentum buffers and optimizer states whose tensor shape equals
    the associated parameter. Unknown parameter-shaped state is safely sliced;
    scalar/non-parameter state is copied as-is.
    """
    if type(old_optimizer) is not type(new_optimizer):
        raise TypeError("Old and new optimizers must have the same type")
    old_named = dict(old_model.named_parameters())
    new_named = dict(new_model.named_parameters())
    if old_named.keys() != new_named.keys():
        missing = sorted(old_named.keys() ^ new_named.keys())
        raise ValueError(f"Compact model parameter names changed: {missing}")
    if len(old_optimizer.param_groups) != len(new_optimizer.param_groups):
        raise ValueError("Optimizer parameter-group count changed during compaction")

    old_names_by_id = {id(param): name for name, param in old_named.items()}
    for old_group, new_group in zip(old_optimizer.param_groups, new_optimizer.param_groups):
        old_group_names = [old_names_by_id.get(id(param)) for param in old_group["params"]]
        new_names_by_id = {id(param): name for name, param in new_named.items()}
        new_group_names = [new_names_by_id.get(id(param)) for param in new_group["params"]]
        if old_group_names != new_group_names:
            raise ValueError("Optimizer parameter order changed during compaction")
        for key, value in old_group.items():
            if key != "params":
                new_group[key] = deepcopy(value)

    for name, old_param in old_named.items():
        new_param = new_named[name]
        state = old_optimizer.state.get(old_param)
        if not state:
            continue
        selectors = parameter_slices.get(name, ())
        copied = {}
        for key, value in state.items():
            copied[key] = _slice_state_tensor(
                value, old_param.shape, new_param.shape, selectors, f"{name}.{key}")
        new_optimizer.state[new_param] = copied
    return new_optimizer


def migrate_scheduler_state(old_scheduler, new_scheduler, old_optimizer, new_optimizer):
    """Preserve the global epoch, milestone counters, and current learning rate."""
    if type(old_scheduler) is not type(new_scheduler):
        raise TypeError("Old and new schedulers must have the same type")
    state = deepcopy(old_scheduler.state_dict())
    new_scheduler.load_state_dict(state)
    if len(old_optimizer.param_groups) != len(new_optimizer.param_groups):
        raise ValueError("Optimizer parameter-group count changed during scheduler migration")
    for old_group, new_group in zip(old_optimizer.param_groups, new_optimizer.param_groups):
        new_group["lr"] = old_group["lr"]
        if "initial_lr" in old_group:
            new_group["initial_lr"] = old_group["initial_lr"]
    return new_scheduler
