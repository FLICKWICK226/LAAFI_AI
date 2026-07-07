from __future__ import annotations

import numpy as np
import torch
from torch import nn


def make_gradcam_overlay(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
) -> np.ndarray:
    """Return a Grad-CAM overlay for one normalized image tensor.

    Requires the optional package installed by `grad-cam`.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as error:
        raise ImportError("Install the optional dependency with: pip install grad-cam") from error

    model.eval()
    device = next(model.parameters()).device
    batch = input_tensor.unsqueeze(0).to(device)
    target = [ClassifierOutputTarget(0)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=batch, targets=target)[0]

    image_np = input_tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = np.clip((image_np * std) + mean, 0, 1)
    return show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
