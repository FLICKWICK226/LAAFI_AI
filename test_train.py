import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from laafi_ai.config import ExperimentConfig
from laafi_ai.data import PCamDataModule
from laafi_ai.model import build_resnet50_classifier
from laafi_ai.trainer import Trainer
import torch

def test():
    config = ExperimentConfig()
    config.data.max_train_samples = 32
    config.data.max_val_samples = 32
    config.data.max_test_samples = 32
    config.training.epochs = 1
    config.training.use_mixup_cutmix = True
    config.data.batch_size = 4

    data_module = PCamDataModule(config.data)
    train_loader, val_loader, _ = data_module.dataloaders()
    model = build_resnet50_classifier(config.model)
    trainer = Trainer(model, config, torch.device("cpu"))
    trainer.fit(train_loader, val_loader)
    print("Test passed without errors!")

if __name__ == "__main__":
    test()
