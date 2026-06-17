from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(slots=True)
class DataConfig:
    dataset_name: str = "1aurent/PatchCamelyon"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    # H&E stain normalization (Macenko)
    use_he_normalization: bool = False
    he_reference_image_path: Optional[str] = None


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 1
    freeze_backbone: bool = True
    unfreeze_layer4: bool = False


@dataclass(slots=True)
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 3
    mixed_precision: bool = True
    use_wandb: bool = False
    wandb_project: str = "LAAFI_AI"
    decision_threshold: float = 0.5
    auto_optimize_threshold: bool = False
    calibrate_probabilities: bool = False
    use_mlflow: bool = False
    mlflow_experiment: str = "LAAFI_AI_PCam"
    mlflow_tracking_uri: str = "mlruns"


@dataclass(slots=True)
class ExperimentConfig:
    project_name: str = "LAAFI_AI_PCam_ResNet50"
    seed: int = 42
    output_dir: str = "outputs"
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        default = cls()
        return cls(
            project_name=raw.get("project_name", default.project_name),
            seed=raw.get("seed", default.seed),
            output_dir=raw.get("output_dir", default.output_dir),
            device=raw.get("device", default.device),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
