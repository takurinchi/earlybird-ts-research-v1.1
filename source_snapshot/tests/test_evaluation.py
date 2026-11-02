import pytest
from evaluation.parameter_count import reduction_rate
from evaluation.classification import classification_metrics


def test_reduction_rate():
    assert reduction_rate(100, 75) == 25
    with pytest.raises(ValueError):
        reduction_rate(0, 0)


def test_classification_metrics():
    metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert metrics["accuracy"] == 0.75
    assert 0 <= metrics["f1_macro"] <= 1
