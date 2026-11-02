"""Non-destructive parameter masks with gradient and post-step enforcement."""
import torch


class MaskManager:
    def __init__(self, model, masks):
        self.model = model
        params = dict(model.named_parameters())
        missing = set(masks) - set(params)
        if missing:
            raise KeyError(f"Masks refer to unknown parameters: {sorted(missing)}")
        self.masks = {}
        for name, mask in masks.items():
            p = params[name]
            if tuple(mask.shape) != tuple(p.shape):
                raise ValueError(f"Mask shape mismatch for {name}")
            self.masks[name] = mask.detach().to(device=p.device, dtype=p.dtype).clone()
            self.masks[name].requires_grad_(False)
            p.register_hook(self._gradient_hook(name))
        self.apply()

    def _gradient_hook(self, name):
        return lambda grad: grad * self.masks[name].to(dtype=grad.dtype)

    @torch.no_grad()
    def apply(self):
        params = dict(self.model.named_parameters())
        for name, mask in self.masks.items():
            params[name].mul_(mask)

    def sparsity(self):
        total = sum(mask.numel() for mask in self.masks.values())
        zeros = sum((mask == 0).sum().item() for mask in self.masks.values())
        return zeros / total if total else 0.0
