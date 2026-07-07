"""Robust checkpoint helpers for restart-safe training.

Saves and restores the full training state so that an interrupted
Colab session can resume exactly where it left off:
model weights, optimizer state, GradScaler state, epoch, best metric,
and training history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from laafi_ai.config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------
# TrainingCheckpoint dataclass
# ------------------------------------------------------------------


@dataclass
class TrainingCheckpoint:
    """Container for a restored training checkpoint.

    Attributes
    ----------
    path : Path
        Filesystem path the checkpoint was loaded from.
    epoch : int
        The epoch that was completed when the checkpoint was saved.
    next_epoch : int
        Convenience: ``epoch + 1`` — the epoch to start from when
        resuming training.
    best_metric : float
        Best validation metric recorded up to the saved epoch.
    history : list[dict[str, float]]
        Training history rows accumulated before the checkpoint.
    config : ExperimentConfig
        The experiment config stored alongside the weights.
    metrics : dict[str, Any]
        Validation metrics snapshot at save time.
    """

    path: Path
    epoch: int
    next_epoch: int
    best_metric: float
    history: list[dict[str, float]] = field(default_factory=list)
    config: ExperimentConfig = field(default_factory=ExperimentConfig)
    metrics: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------


def save_training_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    config: ExperimentConfig,
    epoch: int,
    best_metric: float,
    history: list[dict[str, float]],
    metrics: dict[str, Any] | None = None,
    scaler: Any | None = None,
) -> Path:
    """Persist the full training state to *path*.

    Parameters
    ----------
    path : str | Path
        Destination file path.
    model : nn.Module
        The model whose ``state_dict`` is saved.
    optimizer : Optimizer
        The optimizer whose ``state_dict`` is saved.
    config : ExperimentConfig
        Experiment configuration (serialised to dict).
    epoch : int
        The epoch number that was just completed.
    best_metric : float
        Best validation metric so far.
    history : list[dict]
        Training history rows.
    metrics : dict | None
        Optional metrics snapshot.
    scaler : GradScaler | None
        Optional mixed-precision scaler whose state is saved.

    Returns
    -------
    Path
        The path the checkpoint was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "history": history,
        "metrics": metrics or {},
    }

    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    torch.save(payload, path)
    LOGGER.info("Checkpoint saved → %s  (epoch %d, best_metric=%.4f)",
                path, epoch, best_metric)
    return path


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scaler: Any | None = None,
    device: torch.device | None = None,
) -> TrainingCheckpoint:
    """Restore a training checkpoint saved by :func:`save_training_checkpoint`.

    Parameters
    ----------
    path : str | Path
        Checkpoint file.
    model : nn.Module
        Model whose ``state_dict`` will be loaded **in-place**.
    optimizer : Optimizer | None
        Optimizer whose ``state_dict`` will be loaded **in-place**.
    scaler : GradScaler | None
        Scaler whose ``state_dict`` will be loaded **in-place**.
    device : torch.device | None
        Device to map tensors to.

    Returns
    -------
    TrainingCheckpoint
        Container with epoch, best_metric, history, config, metrics.
    """
    path = Path(path)
    map_location = device if device is not None else "cpu"
    # weights_only=False: checkpoint contains config dict alongside model weights.
    # Only load checkpoints from trusted sources.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", -float("inf"))
    history = ckpt.get("history", [])
    metrics = ckpt.get("metrics", {})
    config = ExperimentConfig.from_dict(ckpt.get("config", {}))

    LOGGER.info("Checkpoint loaded ← %s  (epoch %d, best_metric=%.4f)",
                path, epoch, best_metric)

    return TrainingCheckpoint(
        path=path,
        epoch=epoch,
        next_epoch=epoch + 1,
        best_metric=best_metric,
        history=history,
        config=config,
        metrics=metrics,
    )


# ------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    """Return the most recently modified ``.pt`` file in *directory*.

    Parameters
    ----------
    directory : str | Path
        Directory to scan for ``*.pt`` files.

    Returns
    -------
    Path | None
        Path to the latest checkpoint, or *None* if the directory
        does not exist or contains no ``.pt`` files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None

    candidates = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None

    latest = candidates[-1]
    LOGGER.info("Latest checkpoint found: %s", latest)
    return latest
