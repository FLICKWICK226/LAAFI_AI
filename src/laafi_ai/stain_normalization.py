"""
H&E Stain Normalization using the Macenko method.

This module provides the MacenkoTransform class (PIL → PIL) that normalizes
histopathology images to a reference stain distribution, reducing colour-domain
shift across slides and scanners.

References
----------
Macenko et al., "A method for normalizing histology slides for quantitative
analysis", ISBI 2009. https://doi.org/10.1109/ISBI.2009.5193250

Dependencies
------------
torchstain >= 1.3.0   (pip install torchstain)
Pillow, numpy

Usage
-----
    from laafi_ai.stain_normalization import build_macenko_normalizer

    normalizer = build_macenko_normalizer()          # uses built-in PCam ref
    # or
    normalizer = build_macenko_normalizer("path/to/reference.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic PCam reference image (96×96, RGB uint8, numpy-generated)
# ---------------------------------------------------------------------------
# Generated to approximate the mean H&E appearance of PatchCamelyon slides:
# pinkish eosin background (≈220, 180, 210) with purple hematoxylin nuclei
# (≈130, 90, 160) occupying ~30 % of pixels.
# This avoids shipping a real tissue image in the repo.


def _make_synthetic_pcam_reference() -> Image.Image:
    """Return a synthetic 96×96 PCam-like H&E reference image."""
    rng = np.random.default_rng(42)
    img = np.empty((96, 96, 3), dtype=np.uint8)

    # Eosin background — pinkish
    img[:, :, 0] = rng.integers(200, 235, (96, 96), dtype=np.uint8)
    img[:, :, 1] = rng.integers(165, 195, (96, 96), dtype=np.uint8)
    img[:, :, 2] = rng.integers(195, 225, (96, 96), dtype=np.uint8)

    # Hematoxylin nuclei — purple blobs (~30 % coverage)
    for _ in range(28):
        cy, cx = rng.integers(8, 88, 2)
        r = rng.integers(4, 10)
        yy, xx = np.ogrid[-cy : 96 - cy, -cx : 96 - cx]
        mask = yy**2 + xx**2 <= r**2
        img[mask, 0] = rng.integers(110, 150, mask.sum(), dtype=np.uint8)
        img[mask, 1] = rng.integers(70, 105, mask.sum(), dtype=np.uint8)
        img[mask, 2] = rng.integers(145, 175, mask.sum(), dtype=np.uint8)

    return Image.fromarray(img, mode="RGB")


# ---------------------------------------------------------------------------
# MacenkoTransform
# ---------------------------------------------------------------------------


class MacenkoTransform:
    """PIL-to-PIL H&E normalizer backed by torchstain.

    Parameters
    ----------
    reference_image : PIL.Image.Image | None
        If *None*, the built-in synthetic PCam reference is used.
    luminosity_threshold : float
        Pixels with OD norm below this value are considered background
        and excluded from the stain-matrix estimation.
    alpha : float
        Percentile used for robust singular-value clipping.
    beta : float
        Percentile used for robust min/max estimation.

    Notes
    -----
    * The normalizer is fitted once at construction time on the reference image.
    * If ``torchstain`` is not installed, the transform degrades gracefully
      (identity, logs a warning). This avoids breaking the pipeline in
      environments where the package is missing.
    """

    def __init__(
        self,
        reference_image: Optional[Image.Image] = None,
        luminosity_threshold: float = 0.8,
        alpha: float = 1.0,
        beta: float = 0.15,
    ) -> None:
        self._normalizer = None
        self._enabled = False

        try:
            import torch
            import torchstain  # noqa: F401 — import probe
            from torchstain.normalizers.macenko_normalizer import (
                MacenkoNormalizer as _MacenkoNormalizer,
            )
            from torchvision import transforms as T

            ref_img = (
                reference_image
                if reference_image is not None
                else _make_synthetic_pcam_reference()
            )
            ref_img = ref_img.convert("RGB")

            # torchstain expects a uint8 CHW tensor
            to_tensor = T.ToTensor()  # scales [0,255] → [0,1] float32
            ref_tensor = (to_tensor(ref_img) * 255).to(torch.uint8)  # CHW uint8

            norm = _MacenkoNormalizer()
            norm.fit(ref_tensor)

            self._normalizer = norm
            self._to_tensor = to_tensor
            self._enabled = True
            LOGGER.info(
                "MacenkoTransform initialised successfully on reference (%dx%d).",
                ref_img.width,
                ref_img.height,
            )

        except ImportError:
            LOGGER.warning(
                "torchstain not installed — MacenkoTransform is a no-op. "
                "Install with: pip install torchstain"
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "MacenkoTransform initialisation failed (%s) — falling back to identity.",
                exc,
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True if torchstain is available and the normalizer was fitted."""
        return self._enabled

    def __call__(self, image: Image.Image) -> Image.Image:
        """Normalise *image* and return a new PIL Image.

        If the normalizer is disabled (missing dependency or init error),
        the original image is returned unchanged.
        """
        if not self._enabled:
            return image

        try:
            import torch

            img_rgb = image.convert("RGB")
            img_tensor = (self._to_tensor(img_rgb) * 255).to(torch.uint8)  # CHW uint8
            norm_tensor, _, _ = self._normalizer.normalize(I=img_tensor, stains=True)
            # norm_tensor is float32 in [0,255]; convert back to uint8 PIL
            norm_uint8 = norm_tensor.clamp(0, 255).to(torch.uint8)
            return Image.fromarray(norm_uint8.permute(1, 2, 0).numpy(), mode="RGB")

        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(
                "MacenkoTransform.__call__ failed (%s) — returning original.", exc
            )
            return image

    def __repr__(self) -> str:  # pragma: no cover
        state = "enabled" if self._enabled else "disabled"
        return f"MacenkoTransform({state})"


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_macenko_normalizer(
    reference_image_path: Optional[str] = None,
) -> MacenkoTransform:
    """Construct a :class:`MacenkoTransform`.

    Parameters
    ----------
    reference_image_path : str | None
        Path to a PNG/JPEG reference image.  If *None*, the built-in
        synthetic PCam reference image is used.

    Returns
    -------
    MacenkoTransform
        Ready-to-use callable (PIL → PIL).
    """
    ref_img: Optional[Image.Image] = None
    if reference_image_path is not None:
        path = Path(reference_image_path)
        if path.exists():
            ref_img = Image.open(path).convert("RGB")
            LOGGER.info("Macenko reference loaded from %s", path)
        else:
            LOGGER.warning(
                "Reference image not found at %s — using synthetic PCam reference.",
                path,
            )

    return MacenkoTransform(reference_image=ref_img)
