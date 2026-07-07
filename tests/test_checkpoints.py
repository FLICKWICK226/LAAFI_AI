from pathlib import Path

import torch

from laafi_ai.checkpoints import (
    TrainingCheckpoint,
    find_latest_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from laafi_ai.config import ExperimentConfig


def test_training_checkpoint_roundtrip_restores_resume_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    config = ExperimentConfig(output_dir=str(tmp_path))
    history = [{"epoch": 1.0, "val_auc": 0.75}]

    checkpoint_path = save_training_checkpoint(
        path=tmp_path / "epoch_001.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        best_metric=0.75,
        history=history,
        metrics={"val_auc": 0.75},
    )

    restored = load_training_checkpoint(checkpoint_path, model=model, optimizer=optimizer)

    assert isinstance(restored, TrainingCheckpoint)
    assert restored.path == checkpoint_path
    assert restored.epoch == 1
    assert restored.next_epoch == 2
    assert restored.best_metric == 0.75
    assert restored.history == history
    assert restored.config.output_dir == str(tmp_path)
    assert restored.metrics == {"val_auc": 0.75}


def test_find_latest_checkpoint_uses_modified_time(tmp_path: Path) -> None:
    first = tmp_path / "epoch_001.pt"
    second = tmp_path / "epoch_002.pt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    assert find_latest_checkpoint(tmp_path) == second
    assert find_latest_checkpoint(tmp_path / "missing") is None
