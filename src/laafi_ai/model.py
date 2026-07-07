from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from laafi_ai.config import ModelConfig

LOGGER = logging.getLogger(__name__)


def build_resnet50_classifier(config: ModelConfig) -> nn.Module:
    if config.architecture.lower() != "resnet50":
        raise ValueError(f"Unsupported architecture: {config.architecture}")

    weights = ResNet50_Weights.DEFAULT if config.pretrained else None
    model = resnet50(weights=weights)

    if config.freeze_backbone:
        LOGGER.info("Freezing ResNet50 backbone")
        for parameter in model.parameters():
            parameter.requires_grad = False

    if config.unfreeze_layer4:
        LOGGER.info("Unfreezing ResNet50 layer4")
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.num_classes)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
