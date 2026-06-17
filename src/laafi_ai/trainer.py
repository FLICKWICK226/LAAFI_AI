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

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        best_auc = -np.inf
        checkpoint_dir = self.config.output_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.training.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_metrics = self.evaluate(val_loader)
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_metrics.auc,
                "val_accuracy": val_metrics.accuracy,
                "val_sensitivity": val_metrics.sensitivity,
                "val_specificity": val_metrics.specificity,
            }
            history.append(row)
            LOGGER.info("Epoch %s summary: %s", epoch, row)

            if val_metrics.auc > best_auc:
                best_auc = val_metrics.auc
                self.save_checkpoint(checkpoint_dir / "best_resnet50_pcam.pt", val_metrics)

        return history

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
    def evaluate(self, loader: DataLoader) -> tuple[float, BinaryMetrics]:
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
# pendant la validation
all_logits = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

all_logits = torch.cat(all_logits).numpy()
all_labels = torch.cat(all_labels).numpy()

np.save(output_dir / "val_logits.npy", all_logits)
np.save(output_dir / "val_labels.npy", all_labels)
