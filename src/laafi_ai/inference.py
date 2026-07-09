from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn

from laafi_ai.config import ExperimentConfig
from laafi_ai.data import build_eval_transform
from laafi_ai.model import build_resnet50_classifier


def load_model_from_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[nn.Module, ExperimentConfig]:
    # weights_only=False because checkpoint contains config dict alongside model weights.
    # Only load checkpoints from trusted sources.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    model = build_resnet50_classifier(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def predict_image(
    model: nn.Module,
    image_path: str | Path,
    image_size: int,
    device: torch.device,
) -> float:
    image = Image.open(image_path).convert("RGB")
    transform = build_eval_transform(image_size)
    tensor = transform(image).unsqueeze(0).to(device)
    logit = model(tensor).squeeze()
    return float(torch.sigmoid(logit).cpu())
