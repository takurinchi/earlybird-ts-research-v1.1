"""Persistent BN-output channel gates used while a structural adapter is absent.

Gates model activations, not graph structure; FLOPs and latency must not be
reported as reduced until a structural-pruning backend compacts the model.
"""
import torch


class BNChannelMaskManager:
    def __init__(self, model, masks):
        modules = dict(model.named_modules())
        missing = set(masks) - set(modules)
        if missing:
            raise KeyError(f"Masks refer to unknown modules: {sorted(missing)}")
        self.masks = {}
        self.handles = []
        for name, raw_mask in masks.items():
            module = modules[name]
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                raise TypeError(f"{name} is not a BatchNorm module")
            if raw_mask.numel() != module.num_features:
                raise ValueError(f"Channel count mismatch for {name}")
            mask = raw_mask.detach().bool().to(module.weight.device if module.affine else "cpu")
            self.masks[name] = mask
            self.handles.append(module.register_forward_hook(self._hook(mask)))

    @staticmethod
    def _hook(mask):
        def apply(_module, _inputs, output):
            shape = [1] * output.ndim
            shape[1] = mask.numel()
            return output * mask.to(device=output.device, dtype=output.dtype).view(shape)
        return apply

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def sparsity(self):
        total = sum(mask.numel() for mask in self.masks.values())
        removed = sum((~mask).sum().item() for mask in self.masks.values())
        return removed / total if total else 0.0
