from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def find_optimal_threshold_youden(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    y_true: array de 0/1 (labels réels)
    y_scores: probabilités (sigmoïdes des logits) ou scores continus

    Retourne le seuil qui maximise J = sensibilité + spécificité - 1.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    idx = np.argmax(youden_j)
    return float(thresholds[idx])


def find_threshold_for_target_sensitivity(y_true: np.ndarray, y_scores: np.ndarray, target_sensitivity: float = 0.98) -> float:
    """
    Finds the highest threshold that still guarantees >= target_sensitivity.
    """
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    valid_idx = np.where(recalls >= target_sensitivity)[0]
    if len(valid_idx) == 0:
        return 0.0
    best_idx = valid_idx[-1]
    
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
        
    return float(thresholds[best_idx])

@dataclass(slots=True)
class BinaryMetrics:
    accuracy: float
    auc: float
    average_precision: float
    sensitivity: float
    specificity: float
    precision: float
    # New metrics for target sensitivity evaluation
    threshold_at_98_sens: float = 0.5
    specificity_at_98_sens: float = 0.0
    precision_at_98_sens: float = 0.0


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
    
    # Calculate threshold for 98% sensitivity
    thresh_98 = find_threshold_for_target_sensitivity(labels, probabilities, target_sensitivity=0.98)
    preds_98 = (probabilities >= thresh_98).astype(int)
    tn98, fp98, fn98, tp98 = confusion_matrix(labels, preds_98, labels=[0, 1]).ravel()
    spec_98 = tn98 / (tn98 + fp98) if (tn98 + fp98) > 0 else 0.0
    prec_98 = tp98 / (tp98 + fp98) if (tp98 + fp98) > 0 else 0.0

    return BinaryMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        auc=float(roc_auc_score(labels, probabilities)),
        average_precision=float(average_precision_score(labels, probabilities)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        threshold_at_98_sens=thresh_98,
        specificity_at_98_sens=spec_98,
        precision_at_98_sens=prec_98
    )

def compute_bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Calcule les intervalles de confiance (bootstrap) pour les métriques.
    Retourne un dict: metric_name -> (mean, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    n_samples = len(labels)
    
    bootstrap_metrics = {
        "accuracy": [], "auc": [], "average_precision": [],
        "sensitivity": [], "specificity": [], "precision": [],
        "specificity_at_98_sens": [], "precision_at_98_sens": []
    }
    
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(labels[indices])) < 2:
            continue
            
        m = compute_binary_metrics(labels[indices], probabilities[indices], threshold)
        bootstrap_metrics["accuracy"].append(m.accuracy)
        bootstrap_metrics["auc"].append(m.auc)
        bootstrap_metrics["average_precision"].append(m.average_precision)
        bootstrap_metrics["sensitivity"].append(m.sensitivity)
        bootstrap_metrics["specificity"].append(m.specificity)
        bootstrap_metrics["precision"].append(m.precision)
        bootstrap_metrics["specificity_at_98_sens"].append(m.specificity_at_98_sens)
        bootstrap_metrics["precision_at_98_sens"].append(m.precision_at_98_sens)
        
    results = {}
    for k, v in bootstrap_metrics.items():
        if len(v) > 0:
            mean_val = float(np.mean(v))
            lower_ci = float(np.percentile(v, 2.5))
            upper_ci = float(np.percentile(v, 97.5))
            results[k] = (mean_val, lower_ci, upper_ci)
        else:
            results[k] = (0.0, 0.0, 0.0)
            
    return results
