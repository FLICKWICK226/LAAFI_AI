from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from laafi_ai.config import ExperimentConfig
from laafi_ai.metrics import BinaryMetrics, compute_binary_metrics

LOGGER = logging.getLogger(__name__)


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

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict[str, float]]:
        """Entraîne le modèle. Wrappe automatiquement un run MLflow si use_mlflow=True."""
        if self.config.training.use_mlflow:
            return self._fit_with_mlflow(train_loader, val_loader)
        return self._fit_loop(train_loader, val_loader)

    # ------------------------------------------------------------------
    # MLflow wrapper
    # ------------------------------------------------------------------

    def _fit_with_mlflow(
        self, train_loader: DataLoader, val_loader: DataLoader
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

            history = self._fit_loop(train_loader, val_loader, mlflow_client=mlflow)

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
        mlflow_client=None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        best_auc = -np.inf

        checkpoint_dir = self.config.output_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.training.epochs + 1):
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

            # --- Logger les métriques par epoch dans MLflow ---
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

            if val_metrics.auc > best_auc:
                best_auc = val_metrics.auc
                self.save_checkpoint(checkpoint_dir / "best_resnet50_pcam.pt", val_metrics)

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

    def save_checkpoint(self, path: Path, metrics: BinaryMetrics) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
                "metrics": asdict(metrics),
            },
            path,
        )
        LOGGER.info("Saved checkpoint to %s", path)
