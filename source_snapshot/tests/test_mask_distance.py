import torch
import pytest
from earlybird.mask_distance import mask_distance


def test_identical_masks_have_zero_distance():
    mask = {"layer.weight": torch.tensor([[1, 0], [1, 1]])}
    assert mask_distance(mask, mask) == 0


def test_complementary_masks_have_unit_distance():
    a = {"layer.weight": torch.tensor([1, 0, 1, 0])}
    b = {"layer.weight": torch.tensor([0, 1, 0, 1])}
    assert mask_distance(a, b) == 1


def test_mask_shape_count_mismatch_fails():
    with pytest.raises(ValueError):
        mask_distance(torch.ones(3), torch.ones(4))


def test_layers_with_different_sizes_are_concatenated_correctly():
    a = {"bn_a": torch.tensor([1, 0]), "bn_b": torch.tensor([1, 0, 1])}
    b = {"bn_a": torch.tensor([1, 0]), "bn_b": torch.tensor([1, 1, 0])}
    assert mask_distance(a, b) == pytest.approx(2 / 5)
