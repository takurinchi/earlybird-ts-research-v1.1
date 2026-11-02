import pytest
from pruning.ts_controller import relative_ts_epochs, select_mask


def test_online_uses_only_currently_available_mask():
    current = {"w": 1}
    assert select_mask("online", 3, 3, current_mask=current,
                       available_mask_epoch=3) is current
    with pytest.raises(ValueError):
        select_mask("online", 2, 2, current_mask=current,
                    available_mask_epoch=3)


def test_posthoc_requires_stable_mask():
    stable = {"w": 0}
    assert select_mask("posthoc", 1, 1, stable_mask=stable) is stable
    with pytest.raises(ValueError):
        select_mask("posthoc", 1, 1)


def test_relative_times_use_explicit_offsets():
    assert relative_ts_epochs(10, before=2, near=0, after=3, fixed=5) == {
        "before": 8, "near": 10, "after": 13, "fixed": 5}
