from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import torch

from laafi_ai.config import ExperimentConfig
from laafi_ai.data import PCamDataModule
from laafi_ai.logging_utils import setup_logging
from laafi_ai.model import build_resnet50_classifier, count_trainable_parameters, get_device
from laafi_ai.trainer import Trainer

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet50 on PatchCamelyon.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if args.max_train_samples is not None:
        config.data.max_train_samples = args.max_train_samples
    if args.max_val_samples is not None:
        config.data.max_val_samples = args.max_val_samples
    if args.epochs is not None:
        config.training.epochs = args.epochs

    set_seed(config.seed)
    device = get_device(config.device)
    LOGGER.info("Using device: %s", device)

    data_module = PCamDataModule(config.data)
    train_loader, val_loader, _ = data_module.dataloaders()

    model = build_resnet50_classifier(config.model)
    LOGGER.info("Trainable parameters: %s", count_trainable_parameters(model))

    trainer = Trainer(model=model, config=config, device=device)
    trainer.fit(train_loader, val_loader)

    # --- Semaine 1-2 : Optimisation du seuil de décision (Youden J) ---
    if config.training.auto_optimize_threshold:
        from laafi_ai.threshold_optimization import optimize_threshold_from_val
        result = optimize_threshold_from_val(config.output_path)
        best_threshold = result["threshold"]
        LOGGER.info("Seuil optimal (Youden J) sur val : %.4f", best_threshold)
        config.training.decision_threshold = best_threshold

    # --- Semaine 2-3 : Calibration des probabilités (Temperature Scaling) ---
    if config.training.calibrate_probabilities:
        import numpy as np
        from laafi_ai.calibration import TemperatureScaling, plot_reliability_diagram

        LOGGER.info("Calibration des probabilités par temperature scaling...")
        calibrated = TemperatureScaling(trainer.model).to(device)
        T = calibrated.calibrate(val_loader, device)
        LOGGER.info("Température optimale T=%.4f", T)

        # Reliability diagram avant/après
        val_labels_path = config.output_path / "val_labels.npy"
        val_probs_path = config.output_path / "val_probs.npy"
        if val_labels_path.exists() and val_probs_path.exists():
            val_labels = np.load(val_labels_path)
            val_probs_before = np.load(val_probs_path)
            scaled_logits, _ = calibrated.collect_logits(val_loader, device)
            val_probs_after = torch.sigmoid(
                scaled_logits / calibrated.temperature.cpu()
            ).detach().numpy()
            plot_reliability_diagram(
                val_probs_before,
                val_probs_after,
                val_labels,
                save_path=config.output_path / "reliability_diagram.png",
            )
        else:
            LOGGER.warning(
                "val_labels.npy / val_probs.npy introuvables — "
                "reliability diagram ignoré. Relancer avec save_val_outputs=True."
            )

        # Sauvegarder calibrated_model.pt dans checkpoints/
        checkpoint_dir = config.output_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        calibrated_path = checkpoint_dir / "calibrated_model.pt"
        torch.save(
            {
                "temperature": calibrated.temperature.item(),
                "model_state_dict": calibrated.model.state_dict(),
                "config": config.to_dict(),
            },
            calibrated_path,
        )
        LOGGER.info("Modèle calibré sauvegardé : %s", calibrated_path)


if __name__ == "__main__":
    main()
