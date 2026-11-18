"""Global unstructured magnitude masks; replace with inherited method if different."""
import torch


def global_magnitude_masks(model, sparsity: float, parameter_names=None):
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    allowed = set(parameter_names) if parameter_names else None
    params = [(n, p) for n, p in model.named_parameters()
              if p.requires_grad and (allowed is None or n in allowed)]
    if not params:
        raise ValueError("No eligible parameters found for pruning")
    flat = torch.cat([p.detach().abs().reshape(-1) for _, p in params])
    n_prune = int(sparsity * flat.numel())
    mask_flat = torch.ones_like(flat, dtype=torch.bool)
    if n_prune:
        mask_flat[torch.argsort(flat, stable=True)[:n_prune]] = False
    result, offset = {}, 0
    for name, p in params:
        count = p.numel()
        result[name] = mask_flat[offset:offset + count].reshape_as(p).to(device=p.device)
        offset += count
    return result


def bn_channel_masks(model, pruning_ratio: float):
    """Create Early-Bird channel masks from global |BatchNorm.scale| ranking.

    This returns channel-level selection masks. Turning them into a physically
    narrower network requires a model-specific structural-pruning adapter.
    """
    if not 0 <= pruning_ratio < 1:
        raise ValueError("pruning_ratio must be in [0, 1)")
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
              and module.affine and module.weight is not None]
    if not layers:
        raise ValueError("No affine BatchNorm layers found for channel pruning")
    scores = torch.cat([module.weight.detach().abs().reshape(-1).cpu()
                        for _, module in layers])
    threshold_index = int(scores.numel() * pruning_ratio)
    threshold = torch.sort(scores).values[threshold_index]
    # The official code keeps only scales strictly above the cutoff. Ties at
    # the cutoff can therefore prune slightly more than the requested ratio.
    keep = scores > threshold
    masks, offset = {}, 0
    for name, module in layers:
        size = module.weight.numel()
        masks[name] = keep[offset:offset + size].clone()
        offset += size
    return masks
