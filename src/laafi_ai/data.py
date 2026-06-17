from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from laafi_ai.config import DataConfig

LOGGER = logging.getLogger(__name__)


def build_train_transform(
    image_size: int,
    he_normalizer: Optional[Any] = None,
) -> transforms.Compose:
    """Build the training augmentation pipeline.

    Parameters
    ----------
    image_size : int
        Target spatial size (square crop).
    he_normalizer : MacenkoTransform | None
        Optional H&E stain normalizer applied *before* augmentation.
        When provided, ColorJitter is disabled because colour-based
        augmentations conflict with the normalised stain representation.
    """
    steps: list[Any] = [transforms.Resize((image_size, image_size))]

    if he_normalizer is not None and he_normalizer.enabled:
        # H&E normalisation → deterministic colour space; skip ColorJitter
        steps.append(he_normalizer)
        LOGGER.debug("H&E Macenko normalizer enabled for training transforms (ColorJitter disabled).")
    else:
        # Standard colour augmentation when no stain normalizer is active
        steps.append(
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08)
        )

    steps += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(steps)


def build_eval_transform(
    image_size: int,
    he_normalizer: Optional[Any] = None,
) -> transforms.Compose:
    """Build the evaluation / inference transform pipeline.

    Parameters
    ----------
    image_size : int
        Target spatial size (square crop).
    he_normalizer : MacenkoTransform | None
        Same normalizer used at training time for consistent preprocessing.
    """
    steps: list[Any] = [transforms.Resize((image_size, image_size))]

    if he_normalizer is not None and he_normalizer.enabled:
        steps.append(he_normalizer)
        LOGGER.debug("H&E Macenko normalizer enabled for eval transforms.")

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(steps)


class PCamTorchDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset: Dataset, transform: transforms.Compose) -> None:
        self.dataset = dataset
        self.transform = transform
        self.image_key = "image"
        self.label_key = "label"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item: dict[str, Any] = self.dataset[index]
        image = item[self.image_key]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        label = torch.tensor(float(item[self.label_key]), dtype=torch.float32)
        return self.transform(image), label


@dataclass(slots=True)
class PCamDataModule:
    config: DataConfig

    def load(self) -> DatasetDict:
        LOGGER.info("Loading dataset %s", self.config.dataset_name)
        raw = load_dataset(self.config.dataset_name)
        if not isinstance(raw, DatasetDict):
            raise TypeError("Expected a DatasetDict with train/validation/test splits.")
        return DatasetDict(
            {
                "train": self._limit(self._split(raw, "train"), self.config.max_train_samples),
                "validation": self._limit(
                    self._split(raw, "validation", "valid", "val"),
                    self.config.max_val_samples,
                ),
                "test": self._limit(self._split(raw, "test"), self.config.max_test_samples),
            }
        )

    def dataloaders(
        self,
        he_normalizer: Optional[Any] = None,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train, val, test) DataLoaders.

        Parameters
        ----------
        he_normalizer : MacenkoTransform | None
            When provided, stain normalization is applied to all splits.
        """
        raw = self.load()
        train_ds = PCamTorchDataset(
            raw["train"],
            build_train_transform(self.config.image_size, he_normalizer=he_normalizer),
        )
        val_ds = PCamTorchDataset(
            raw["validation"],
            build_eval_transform(self.config.image_size, he_normalizer=he_normalizer),
        )
        test_ds = PCamTorchDataset(
            raw["test"],
            build_eval_transform(self.config.image_size, he_normalizer=he_normalizer),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, val_loader, test_loader

    @staticmethod
    def _limit(dataset: Dataset, max_samples: int | None) -> Dataset:
        if max_samples is None:
            return dataset
        max_samples = min(max_samples, len(dataset))
        return dataset.select(range(max_samples))

    @staticmethod
    def _split(dataset_dict: DatasetDict, *names: str) -> Dataset:
        for name in names:
            if name in dataset_dict:
                return dataset_dict[name]
        available = ", ".join(dataset_dict.keys())
        requested = ", ".join(names)
        raise KeyError(f"Missing split. Requested one of [{requested}], available: [{available}]")
