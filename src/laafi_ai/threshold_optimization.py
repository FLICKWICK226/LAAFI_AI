import numpy as np
from pathlib import Path
from .metrics import find_optimal_threshold_youden, compute_metrics  # à adapter

def optimize_threshold_from_val(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    val_logits = np.load(output_dir / "val_logits.npy")
    val_labels = np.load(output_dir / "val_labels.npy")

    # logits -> probabilités
    val_probs = 1 / (1 + np.exp(-val_logits))

    best_threshold = find_optimal_threshold_youden(val_labels, val_probs)

    # recalcul des métriques validation avec ce seuil
    val_preds = (val_probs >= best_threshold).astype(int)
    metrics_val = compute_metrics(val_labels, val_preds, val_probs)

    return {
        "threshold": best_threshold,
        "val_metrics": metrics_val,
    }
