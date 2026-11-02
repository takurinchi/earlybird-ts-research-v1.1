import torch
import pytest
from earlybird.stability_detector import StabilityDetector


def test_official_window_requires_five_masks_and_detects_first_stable_window():
    detector = StabilityDetector(threshold=0.1, mask_window=5)
    masks = [{"bn.weight": torch.tensor([1, 0, 1])} for _ in range(5)]
    for epoch, mask in enumerate(masks[:4]):
        assert not detector.update(epoch, mask)
    assert detector.update(4, masks[4])
    assert detector.tm == 4
    assert len(detector.last_distances) == 4
    assert not detector.update(5, masks[4])
    assert len(detector.last_distances) == 4


def test_distance_equal_to_threshold_is_stable_like_reference_code():
    detector = StabilityDetector(threshold=0.5, mask_window=2)
    assert not detector.update(0, torch.tensor([0, 0]))
    assert detector.update(1, torch.tensor([1, 0]))


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        StabilityDetector(threshold=-1)
