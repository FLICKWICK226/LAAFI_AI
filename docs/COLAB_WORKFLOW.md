# LAAFI_AI — Colab Workflow Guide

## Overview

The LAAFI_AI workflow is split into **five focused notebooks** that can each be run **top-to-bottom after a fresh runtime restart**. No notebook depends on hidden variables from another notebook.

## Notebook Sequence

| # | Notebook | Purpose |
|---|----------|---------|
| 00 | `00_setup.ipynb` | Mount Drive, install deps, verify imports, resolve paths |
| 01 | `01_data_prep.ipynb` | Load PatchCamelyon, inspect splits, visualise samples |
| 02 | `02_train.ipynb` | Train or **resume** training from checkpoint |
| 03 | `03_eval.ipynb` | Evaluate best checkpoint on test set, generate report |
| 04 | `04_inference.ipynb` | Single-image predictions + Grad-CAM interpretability |

## How To Run

1. Open **`00_setup.ipynb`** in Google Colab. Run all cells to mount Drive and install dependencies.
2. Open **`01_data_prep.ipynb`**. Run all cells to download the dataset and verify it loads correctly.
3. Open **`02_train.ipynb`**. Adjust the config (smoke test vs. full training) and run all cells. Checkpoints are saved every epoch to `outputs_finetune_layer4/checkpoints/`.
4. Open **`03_eval.ipynb`**. Run all cells. It loads `best_resnet50_pcam.pt` and generates ROC, PR, confusion matrix, and a metrics CSV.
5. Open **`04_inference.ipynb`**. Run all cells. It loads the best checkpoint for predictions and Grad-CAM overlays.

## What Happens If Colab Disconnects?

**Nothing is lost.** Every epoch saves a full checkpoint containing:

- Model weights
- Optimizer state
- GradScaler state
- Epoch number
- Best validation AUC
- Training history

When you reopen `02_train.ipynb` and run it from the top, the Trainer **automatically detects** the latest `epoch_NNN.pt` checkpoint and resumes training from the next epoch.

## Where Are Outputs Stored?

All artifacts are written under the configured `output_dir` (default: `outputs_finetune_layer4/`):

```
outputs_finetune_layer4/
├── checkpoints/
│   ├── epoch_001.pt
│   ├── epoch_002.pt
│   ├── ...
│   └── best_resnet50_pcam.pt
├── figures/
│   ├── training_curves.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── confusion_matrix.png
│   └── gradcam_samples.png
├── metrics/
│   └── metrics_finales.csv
├── predictions/
│   ├── test_labels.npy
│   └── test_probs.npy
├── reports/
├── val_labels.npy
└── val_probs.npy
```

## Key Design Principles

1. **No hidden runtime state.** Every notebook reloads config and checkpoints from disk.
2. **Restart-safe training.** Full optimizer + scaler + epoch state is saved every epoch.
3. **Independent notebooks.** Evaluation and inference work in a fresh session without rerunning training.
4. **Explicit paths.** All outputs go to named directories — no reliance on implicit variables.
5. **Beginner-friendly.** French markdown headers and comments explain each step.

## Smoke Test vs. Full Training

In `02_train.ipynb`, uncomment these lines for a quick smoke test:

```python
config.data.max_train_samples = 512
config.data.max_val_samples = 256
```

Comment them out (or set to `None`) for a full training run on the entire PatchCamelyon dataset.
