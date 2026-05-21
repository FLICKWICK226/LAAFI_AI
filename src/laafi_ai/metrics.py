from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(slots=True)
class BinaryMetrics:
    accuracy: float
    auc: float
    average_precision: float
    sensitivity: float
    specificity: float
    precision: float


def compute_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    labels = labels.astype(int)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return BinaryMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        auc=float(roc_auc_score(labels, probabilities)),
        average_precision=float(average_precision_score(labels, probabilities)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision_score(labels, predictions, zero_division=0)),
    )
