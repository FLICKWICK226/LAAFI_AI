from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from laafi_ai.checkpoints import (
    TrainingCheckpoint,
    find_latest_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from laafi_ai.config import ExperimentConfig
from laafi_ai.metrics import BinaryMetrics, compute_binary_metrics

LOGGER = logging.getLogger(__name__)


def rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def apply_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def apply_cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def _try_import_mlflow():
    """Import mlflow paresseusement — pas de crash si non installé."""
    try:
        import mlflow
        return mlflow
    except ImportError:
        LOGGER.warning(
            "mlflow non installé. Désactiver use_mlflow ou pip install mlflow."
        )
        return None


class Trainer:
    def __init__(self, model: nn.Module, config: ExperimentConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=config.training.mixed_precision and device.type == "cuda"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_checkpoint: TrainingCheckpoint | None = None,
    ) -> list[dict[str, float]]:
        """Entraîne le modèle. Wrappe automatiquement un run MLflow si use_mlflow=True.

        Parameters
        ----------
        resume_checkpoint : TrainingCheckpoint | None
            When provided, training resumes from the given epoch.
            Model/optimizer states must already be loaded before
            calling :meth:`fit`.
        """
        if self.config.training.use_mlflow:
            return self._fit_with_mlflow(train_loader, val_loader, resume_checkpoint)
        return self._fit_loop(train_loader, val_loader, resume_checkpoint=resume_checkpoint)

    # ------------------------------------------------------------------
    # MLflow wrapper
    # ------------------------------------------------------------------

    def _fit_with_mlflow(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_checkpoint: TrainingCheckpoint | None = None,
    ) -> list[dict[str, float]]:
        mlflow = _try_import_mlflow()
        if mlflow is None:
            LOGGER.warning("Fallback: entraînement sans MLflow.")
            return self._fit_loop(train_loader, val_loader)

        mlflow.set_tracking_uri(self.config.training.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.training.mlflow_experiment)

        with mlflow.start_run(run_name=self.config.project_name):
            # --- Logger tous les hyperparamètres ---
            mlflow.log_params({
                # Optimizer
                "lr": self.config.optimizer.learning_rate,
                "weight_decay": self.config.optimizer.weight_decay,
                # Training
                "epochs": self.config.training.epochs,
                "decision_threshold": self.config.training.decision_threshold,
                "mixed_precision": self.config.training.mixed_precision,
                # Model
                "architecture": self.config.model.architecture,
                "pretrained": self.config.model.pretrained,
                "freeze_backbone": self.config.model.freeze_backbone,
                "unfreeze_layer4": self.config.model.unfreeze_layer4,
                # Data
                "batch_size": self.config.data.batch_size,
                "image_size": self.config.data.image_size,
                "dataset": self.config.data.dataset_name,
            })

            history = self._fit_loop(train_loader, val_loader, mlflow_client=mlflow, resume_checkpoint=resume_checkpoint)

            # --- Logger le best checkpoint comme artifact ---
            best_ckpt = self.config.output_path / "checkpoints" / "best_resnet50_pcam.pt"
            if best_ckpt.exists():
                mlflow.log_artifact(str(best_ckpt), artifact_path="checkpoints")
                LOGGER.info("MLflow artifact loggé : %s", best_ckpt)

            # --- Logger les métriques finales (best epoch) ---
            if history:
                best_row = max(history, key=lambda r: r["val_auc"])
                mlflow.log_metrics({
                    "best_val_auc": best_row["val_auc"],
                    "best_val_sensitivity": best_row["val_sensitivity"],
                    "best_val_specificity": best_row["val_specificity"],
                    "best_val_accuracy": best_row["val_accuracy"],
                    "best_epoch": best_row["epoch"],
                })

        return history

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _fit_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mlflow_client: Any | None = None,
        resume_checkpoint: TrainingCheckpoint | None = None,
    ) -> list[dict[str, float]]:
        # Resume state if provided, otherwise start fresh.
        if resume_checkpoint is not None:
            start_epoch = resume_checkpoint.next_epoch
            history: list[dict[str, float]] = list(resume_checkpoint.history)
            best_auc = resume_checkpoint.best_metric
            LOGGER.info("Resuming training from epoch %d (best AUC so far: %.4f)",
                        start_epoch, best_auc)
        else:
            start_epoch = 1
            history = []
            best_auc = -np.inf

        checkpoint_dir = self.config.output_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, self.config.training.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_metrics = self.evaluate(val_loader, save_val_outputs=True)

            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_metrics.auc,
                "val_accuracy": val_metrics.accuracy,
                "val_sensitivity": val_metrics.sensitivity,
                "val_specificity": val_metrics.specificity,
                "val_precision": val_metrics.precision,
                "val_average_precision": val_metrics.average_precision,
            }
            history.append(row)
            LOGGER.info("Epoch %s summary: %s", epoch, row)

            # --- Log per-epoch metrics to MLflow ---
            if mlflow_client is not None:
                mlflow_client.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_auc": val_metrics.auc,
                        "val_accuracy": val_metrics.accuracy,
                        "val_sensitivity": val_metrics.sensitivity,
                        "val_specificity": val_metrics.specificity,
                        "val_precision": val_metrics.precision,
                        "val_average_precision": val_metrics.average_precision,
                    },
                    step=epoch,
                )

            is_best = val_metrics.auc > best_auc
            if is_best:
                best_auc = val_metrics.auc

            # Always save an epoch checkpoint (for resume) …
            save_training_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model=self.model,
                optimizer=self.optimizer,
                config=self.config,
                epoch=epoch,
                best_metric=best_auc,
                history=history,
                metrics=asdict(val_metrics),
                scaler=self.scaler,
            )

            # … and keep the "best" link updated.
            if is_best:
                save_training_checkpoint(
                    path=checkpoint_dir / "best_resnet50_pcam.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    config=self.config,
                    epoch=epoch,
                    best_metric=best_auc,
                    history=history,
                    metrics=asdict(val_metrics),
                    scaler=self.scaler,
                )

        return history

    # ------------------------------------------------------------------
    # Per-epoch methods (inchangés)
    # ------------------------------------------------------------------

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        losses: list[float] = []

        progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
        for images, labels in progress:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.config.training.use_mixup_cutmix:
                if np.random.rand() < 0.5:
                    images, labels = apply_mixup(images, labels, self.config.training.mixup_alpha)
                else:
                    images, labels = apply_cutmix(images, labels, self.config.training.cutmix_alpha)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=self.config.training.mixed_precision and self.device.type == "cuda"
            ):
                logits = self.model(images).squeeze(1)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=np.mean(losses))

        return float(np.mean(losses))

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        save_val_outputs: bool = False,
    ) -> tuple[float, BinaryMetrics]:
        self.model.eval()
        losses: list[float] = []
        labels_all: list[np.ndarray] = []
        probabilities_all: list[np.ndarray] = []

        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images).squeeze(1)
            loss = self.criterion(logits, labels)
            probabilities = torch.sigmoid(logits)

            losses.append(float(loss.cpu()))
            labels_all.append(labels.cpu().numpy())
            probabilities_all.append(probabilities.cpu().numpy())

        labels_np = np.concatenate(labels_all)
        probabilities_np = np.concatenate(probabilities_all)

        metrics = compute_binary_metrics(
            labels_np,
            probabilities_np,
            threshold=self.config.training.decision_threshold,
        )

        if save_val_outputs:
            output_dir = self.config.output_path
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "val_labels.npy", labels_np)
            np.save(output_dir / "val_probs.npy", probabilities_np)

        return float(np.mean(losses)), metrics

    def resume_from_checkpoint(
        self,
        checkpoint_dir: str | Path,
    ) -> TrainingCheckpoint | None:
        """Attempt to find and load the latest epoch checkpoint.

        Restores model weights, optimizer state, and scaler state
        **in-place** on this Trainer instance.

        Returns
        -------
        TrainingCheckpoint | None
            The restored checkpoint, or *None* if no checkpoint was found.
        """
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest is None:
            LOGGER.info("No checkpoint found in %s — training from scratch.", checkpoint_dir)
            return None

        return load_training_checkpoint(
            latest,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            device=self.device,
        )
