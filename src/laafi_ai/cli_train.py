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
    if cfg.training.auto_optimize_threshold:
        result = optimize_threshold_from_val(cfg.paths.output_dir)
        best_threshold = result["threshold"]
        logger.info(f"Optimal threshold (Youden J) on val: {best_threshold:.4f}")


if __name__ == "__main__":
    main()
