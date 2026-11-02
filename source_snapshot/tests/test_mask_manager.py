import torch
import pytest
from pruning.mask_manager import MaskManager


def test_mask_stays_applied_after_optimizer_step():
    model = torch.nn.Linear(2, 1, bias=False)
    masks = {"weight": torch.tensor([[1.0, 0.0]])}
    manager = MaskManager(model, masks)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, weight_decay=0.1)
    model(torch.ones(1, 2)).sum().backward(); optimizer.step(); manager.apply()
    assert model.weight[0, 1].item() == 0
    assert manager.sparsity() == 0.5


def test_shape_mismatch_fails():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError):
        MaskManager(model, {"weight": torch.ones(5)})
