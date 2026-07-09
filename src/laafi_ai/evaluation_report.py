"""Evaluation reporting helpers for LAAFI_AI.

Generates publication-ready evaluation plots (ROC curve, Precision-Recall
curve, confusion matrix) and CSV metric tables.  All functions accept
numpy arrays and output paths so they work identically from a fresh
Colab restart — no hidden runtime state required.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from laafi_ai.metrics import BinaryMetrics, compute_bootstrap_metrics

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------
# ROC curve
# ------------------------------------------------------------------


def plot_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    save_path: str | Path,
) -> None:
    """Plot and save a ROC curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc_score = roc_auc_score(labels, probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info("ROC curve saved → %s", save_path)


# ------------------------------------------------------------------
# Precision–Recall curve
# ------------------------------------------------------------------


def plot_pr_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    save_path: str | Path,
) -> None:
    """Plot and save a Precision–Recall curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    ap = average_precision_score(labels, probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.4f}", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info("PR curve saved → %s", save_path)


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str | Path,
) -> None:
    """Plot and save a confusion matrix."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Métastase"]).plot(ax=ax)
    ax.set_title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info("Confusion matrix saved → %s", save_path)


# ------------------------------------------------------------------
# Metrics CSV
# ------------------------------------------------------------------


def save_metrics_csv(
    metrics: BinaryMetrics,
    save_path: str | Path,
    threshold: float = 0.5,
    bootstrap_results: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Write a single-row CSV summary of evaluation metrics."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "threshold",
        "accuracy",
        "auc",
        "average_precision",
        "sensitivity",
        "specificity",
        "precision",
    ]

    row = {"threshold": f"{threshold:.4f}"}

    if bootstrap_results:
        for k in [
            "accuracy",
            "auc",
            "average_precision",
            "sensitivity",
            "specificity",
            "precision",
        ]:
            mean_v, low_v, high_v = bootstrap_results[k]
            row[k] = f"{mean_v:.4f} [{low_v:.4f}, {high_v:.4f}]"
    else:
        row["accuracy"] = f"{metrics.accuracy:.4f}"
        row["auc"] = f"{metrics.auc:.4f}"
        row["average_precision"] = f"{metrics.average_precision:.4f}"
        row["sensitivity"] = f"{metrics.sensitivity:.4f}"
        row["specificity"] = f"{metrics.specificity:.4f}"
        row["precision"] = f"{metrics.precision:.4f}"

    with open(save_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    LOGGER.info("Metrics CSV saved → %s", save_path)


# ------------------------------------------------------------------
# Full evaluation report (convenience wrapper)
# ------------------------------------------------------------------


def generate_full_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metrics: BinaryMetrics,
    figures_dir: str | Path,
    metrics_dir: str | Path,
    threshold: float = 0.5,
) -> None:
    """Generate all standard evaluation artifacts at once.

    Produces ROC curve, PR curve, confusion matrix, and metrics CSV.

    Parameters
    ----------
    labels : np.ndarray
        Ground-truth binary labels.
    probabilities : np.ndarray
        Model output probabilities.
    metrics : BinaryMetrics
        Pre-computed metrics dataclass.
    figures_dir : Path-like
        Directory for figure outputs.
    metrics_dir : Path-like
        Directory for CSV outputs.
    threshold : float
        Decision threshold used for binary predictions.
    """
    figures_dir = Path(figures_dir)
    metrics_dir = Path(metrics_dir)

    predictions = (probabilities >= threshold).astype(int)

    LOGGER.info("Computing bootstrap confidence intervals (this may take a moment)...")
    bootstrap_results = compute_bootstrap_metrics(
        labels, probabilities, threshold=threshold
    )

    plot_roc_curve(labels, probabilities, figures_dir / "roc_curve.png")
    plot_pr_curve(labels, probabilities, figures_dir / "pr_curve.png")
    plot_confusion_matrix(labels, predictions, figures_dir / "confusion_matrix.png")
    save_metrics_csv(
        metrics,
        metrics_dir / "metrics_finales.csv",
        threshold=threshold,
        bootstrap_results=bootstrap_results,
    )

    LOGGER.info("Full evaluation report generated.")
