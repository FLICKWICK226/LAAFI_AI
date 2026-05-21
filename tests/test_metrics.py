import numpy as np

from laafi_ai.metrics import compute_binary_metrics


def test_binary_metrics_perfect_predictions() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.2, 0.8, 0.95])
    metrics = compute_binary_metrics(labels, probabilities)

    assert metrics.accuracy == 1.0
    assert metrics.auc == 1.0
    assert metrics.sensitivity == 1.0
    assert metrics.specificity == 1.0
