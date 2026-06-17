from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import compute_binary_metrics, find_optimal_threshold_youden


def optimize_threshold_from_val(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)

    val_labels = np.load(output_dir / "val_labels.npy")
    val_probs = np.load(output_dir / "val_probs.npy")

    best_threshold = find_optimal_threshold_youden(val_labels, val_probs)

    val_metrics = compute_binary_metrics(
        val_labels,
        val_probs,
        threshold=best_threshold,
    )

    return {
        "threshold": best_threshold,
        "val_metrics": val_metrics,
    }
