from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """Wrapper post-hoc de calibration des probabilités par temperature scaling.

    Le modèle encapsulé est mis en eval() et ses paramètres sont freezés.
    Seul le scalaire T est optimisé par LBFGS sur la NLL du validation set.

    T > 1 → modèle over-confident → calibration adoucit les probabilités.
    T < 1 → modèle under-confident → rare en pratique.
    T = 1 → aucun changement (état initial).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature

    @torch.no_grad()
    def collect_logits(
        self,
        loader: DataLoader,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collecte les logits bruts (avant sigmoid) et les labels sur un loader."""
        self.model.eval()
        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        for images, labels in loader:
            images = images.to(device)
            logits = self.model(images).squeeze(1)
            logits_list.append(logits.cpu())
            labels_list.append(labels.float().cpu())
        return torch.cat(logits_list), torch.cat(labels_list)

    def calibrate(
        self,
        val_loader: DataLoader,
        device: torch.device,
        max_iter: int = 50,
    ) -> float:
        """Optimise T par LBFGS sur la NLL du validation set.

        Tous les paramètres du backbone sont freezés — seul T est libre.
        Retourne la valeur optimale de T.
        """
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.temperature.requires_grad_(True)

        logits, labels = self.collect_logits(val_loader, device)
        logits = logits.to(device)
        labels = labels.to(device)

        nll_criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        T_val = float(self.temperature.item())
        LOGGER.info("Temperature scaling optimisé : T=%.4f", T_val)
        return T_val


def plot_reliability_diagram(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path = "outputs/reliability_diagram.png",
    n_bins: int = 10,
) -> None:
    """Trace et sauvegarde le reliability diagram avant/après calibration."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Reliability Diagram — Calibration des probabilités", fontsize=13)

    for ax, probs, title in [
        (axes[0], probs_before, "Avant calibration (T=1.0)"),
        (axes[1], probs_after, "Après calibration (Temperature Scaling)"),
    ]:
        frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, "s-", label="Modèle", color="#1f77b4")
        ax.plot([0, 1], [0, 1], "k--", label="Calibration parfaite", linewidth=1.5)
        ax.set_xlabel("Probabilité prédite (moyenne par bin)")
        ax.set_ylabel("Fraction de positifs réels")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info("Reliability diagram sauvegardé : %s", save_path)
