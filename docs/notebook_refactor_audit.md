# Notebook Refactor Audit

## Scope

This audit covers the current notebook-based workflow for LAAFI_AI, especially:

- `notebooks/00_explication_debutant_pipeline.ipynb`
- `notebooks/01_colab_training_resnet50_pcam.ipynb`
- existing reusable code under `src/laafi_ai/`
- current API/UI inference prototypes under `src/`

The goal is to identify the structural issues that make the workflow fragile after a Google Colab restart, then guide a modular, restart-safe refactor.

## 1. Monolithic Notebooks

### `00_explication_debutant_pipeline.ipynb`

This notebook is short, educational, and mostly safe. It explains the project flow and shows simple metric plots. It does not run heavyweight training.

### `01_colab_training_resnet50_pcam.ipynb`

This notebook is monolithic. It mixes:

- Colab setup and Drive mounting
- dependency installation
- path setup
- config mutation
- data loading
- preview visualization
- training
- checkpoint lookup
- repeated training attempts
- final test evaluation
- metric table generation
- report generation
- Grad-CAM visualization
- qualitative analysis

This makes it hard to resume after runtime interruption because later cells assume earlier variables are still alive.

## 2. Unsafe Runtime-State Dependencies

The training notebook creates and reuses many variables in memory:

- `PROJECT_ROOT`
- `config`
- `device`
- `data_module`
- `train_loader`
- `val_loader`
- `test_loader`
- `images`
- `labels`
- `model`
- `trainer`
- `history`
- `history_feature_extraction`
- `model_eval`
- `all_probs`
- `all_preds`
- `all_labels`
- `target_layers`
- helper functions such as `denormalize_to_rgb`

After a Colab runtime restart, evaluation and Grad-CAM cells can fail unless all previous cells are rerun in exactly the expected order. This violates the target principle: every later notebook must be runnable after a fresh runtime restart.

## 3. Logic That Should Move Into Python Modules

The following notebook logic should be extracted into reusable modules:

- project path and artifact path resolution
- Colab/local environment detection
- checkpoint discovery
- checkpoint saving/loading for training resume
- training-state resume logic
- history persistence
- final test evaluation and metric CSV writing
- ROC/PR/confusion matrix figure generation
- report generation
- Grad-CAM sample collection and export
- inference loading from saved checkpoint

Some reusable logic already exists in `src/laafi_ai/`, including data loading, model creation, training, metrics, evaluation, inference, calibration, and Grad-CAM helpers. The refactor should extend these modules rather than duplicating notebook code.

## 4. Current Checkpoint State

Current training checkpoints are saved by `Trainer.save_checkpoint()` with:

- `model_state_dict`
- `config`
- `metrics`

This is enough for model evaluation, but not enough for robust training resume. A restart-safe training checkpoint should also save:

- optimizer state
- scheduler state, if used
- epoch number
- best metric
- training history

The notebook has a `find_latest_checkpoint()` helper, but it lives only inside the notebook and loads model weights without restoring optimizer state or epoch progress.

## 5. Path Problems

The notebook hardcodes Colab-specific paths such as:

- `/content/drive/MyDrive/LAAFI_AI`

The API/UI prototypes hardcode:

- `../resnet50_finetuned_weights.pth`

The workflow needs a central path configuration that supports both:

- Google Colab with Google Drive persistence
- local Jupyter or local script execution

All persistent artifacts should be written under explicit directories such as:

- checkpoints
- metrics
- figures
- reports
- predictions

## 6. Cells That Do Too Much

Large cells currently combine unrelated responsibilities. Examples:

- final evaluation loads a checkpoint, runs test inference, computes metrics, saves CSV, and generates three figures in one cell
- Grad-CAM cells search examples, compute activations, save images, and display plots together
- training cells mutate config, create model/trainer, search checkpoint, load checkpoint, and train

These should become small notebook calls into named functions.

## 7. Execution Order Risks

The current notebook execution order is unsafe:

- evaluation depends on `config`, `device`, `test_loader`, and `model_eval`
- Grad-CAM depends on `model_eval`, `test_loader`, `device`, and `target_layers`
- reporting depends on `outputs_final/metrics_finales.csv`
- training resume depends on notebook-local `find_latest_checkpoint()`

These dependencies should be made explicit through saved artifacts and reload functions.

## 8. Existing Strengths To Preserve

The project already has a good foundation:

- `ExperimentConfig` dataclasses and YAML loading
- modular data transforms and `PCamDataModule`
- modular ResNet50 builder
- `Trainer` class
- reusable metric functions
- checkpoint format that already includes config and model weights
- final portfolio outputs already saved under `outputs_final/`
- clear beginner-oriented French explanations

The refactor should preserve behavior while improving restart safety.

## 9. Proposed Restructuring

Create focused notebooks:

- `00_setup.ipynb`: setup, Drive/local paths, imports, config, device
- `01_data_prep.ipynb`: dataset checks and DataLoader smoke check
- `02_train.ipynb`: training from scratch or resume from checkpoint
- `03_eval.ipynb`: load checkpoint and evaluate independently
- `04_inference.ipynb`: load checkpoint and run predictions independently

Add reusable helpers:

- `src/laafi_ai/paths.py`
- `src/laafi_ai/checkpoints.py`
- `src/laafi_ai/evaluation_report.py`

Update training checkpoints so they can resume full training state.

## 10. Risks And Assumptions

- The full PatchCamelyon dataset is large and may not be available during local verification.
- Full training cannot be practically verified inside a short local coding session.
- Notebook generation should avoid adding unnecessary dependencies.
- Existing output files should not be deleted.
- Existing beginner explanations should be preserved or moved into documentation.

## 11. Next Step

Implement Phase 1:

1. add path helpers for local/Colab artifact roots,
2. add restart-safe checkpoint helpers with tests,
3. adapt the trainer to save and resume full training state,
4. rewrite notebooks so each one can run top-to-bottom after a fresh restart.
