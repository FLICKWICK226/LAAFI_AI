from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from laafi_ai.metrics import BinaryMetrics, compute_binary_metrics


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    for images, labels in tqdm(loader, desc="predict", leave=False):
        images = images.to(device)
        logits = model(images).squeeze(1)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        labels_all.append(labels.numpy())
        probabilities_all.append(probabilities)
    return np.concatenate(labels_all), np.concatenate(probabilities_all)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> BinaryMetrics:
    labels, probabilities = collect_predictions(model, loader, device)
    return compute_binary_metrics(labels, probabilities, threshold)
