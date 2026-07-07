# AGENTS.md — Refactor Notebook/Colab Workflow for LAAFI_AI

## Mission

You are a coding agent working inside the `LAAFI_AI` project
.

Your mission is to transform the current notebook-based workflow from a fragile, monolithic, state-dependent Google Colab notebook into a reproducible, modular, restart-safe workflow that works well in Google Colab and GitHub.

The current problem is:
- the training notebook is a single massive block,
- Google Colab free compute units are exhausted during training,
- after runtime disconnection or restart, later cells fail because variables are no longer defined,
- the workflow depends too much on in-memory state instead of saved artifacts and checkpoints.

Your job is to restructure the workflow so that training, evaluation, and inference can resume safely after a runtime interruption.

---

## Context

This folder
 contains an AI/ML project named **LAAFI_AI**.
The user is still relatively new to notebook environments and needs:
- clean notebook structure,
- checkpoint/resume support,
- minimal dependence on hidden runtime state,
- reproducible execution from top to bottom,
- practical usability inside Google Colab.

The user is not asking for a theoretical notebook tutorial.
The user wants the repo changed into something robust and executable.

---

## Main objectives

1. Audit the current notebook(s) and identify structural problems.
2. Break the monolithic notebook into smaller logical units.
3. Extract reusable logic from notebook cells into Python modules.
4. Make every important stage restart-safe.
5. Ensure training can resume from checkpoints.
6. Ensure evaluation/inference notebooks reload saved artifacts instead of relying on RAM variables.
7. Improve readability for a beginner notebook user.
8. Document exactly how to use the new workflow in Colab.

---

## Required design principles

Follow these principles strictly:

- Never rely on notebook runtime state across sessions.
- Any later notebook must be able to run after a fresh runtime restart.
- Long repeated code must be extracted into `.py` modules.
- Keep notebooks short and task-focused.
- Prefer explicit file-based inputs/outputs over implicit variables.
- Every notebook must be runnable top-to-bottom.
- Every notebook must begin with setup/config cells.
- Store all important artifacts in persistent storage paths, suitable for Google Drive.
- Use checkpoints frequently during training.
- Save enough state to resume training correctly, including model state, optimizer state, epoch/step, and relevant metrics when applicable. [web:80][web:85]
- Add markdown headings and comments so the workflow is understandable for a beginner. [web:75][web:87]
- Avoid giant cells; split large logic into functions/modules. [web:87][web:102]
- Favor reproducibility over cleverness. [web:87][web:95][web:98]

---

## Target architecture

Unless project
 constraints force a better structure, aim for something close to this:

```text
LAAFI_AI/
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── 01_data_prep.ipynb
│   ├── 02_train.ipynb
│   ├── 03_eval.ipynb
│   └── 04_inference.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── train.py
│   ├── eval.py
│   ├── inference.py
│   ├── checkpoints.py
│   └── utils.py
├── checkpoints/
├── outputs/
├── docs/
│   └── COLAB_WORKFLOW.md
└── AGENTS.md
```

If the existing project
 structure suggests a better naming convention, adapt it, but keep the same separation of concerns.

---

## Notebook responsibilities

### `00_setup.ipynb`
Must contain only:
- environment setup,
- package checks/installs if necessary,
- Google Drive mount instructions if relevant,
- import verification,
- path configuration,
- seed/config loading,
- device detection.

### `01_data_prep.ipynb`
Must contain only:
- dataset loading,
- cleaning,
- split creation,
- preprocessing,
- saving prepared artifacts to disk.

### `02_train.ipynb`
Must contain only:
- loading prepared artifacts,
- model creation/loading,
- optimizer/scheduler setup,
- training loop,
- checkpoint saving,
- resume-from-checkpoint logic,
- training metrics persistence.

### `03_eval.ipynb`
Must:
- reload model from saved checkpoint,
- reload validation/test artifacts from disk,
- run evaluation independently of previous notebook state,
- save metrics/results.

### `04_inference.ipynb`
Must:
- load final/best checkpoint from disk,
- run prediction independently,
- save outputs/visualizations/results cleanly.

---

## What to audit first

Before changing code, inspect the project
 and identify:

1. Which notebook(s) are monolithic.
2. Which variables are created in one place and reused later without persistence.
3. Which sections should become Python modules.
4. Where checkpoints are already used, if at all.
5. Whether training currently saves only weights or also optimizer/training state.
6. Which paths are local-runtime-only and should be replaced with persistent paths.
7. Which cells are too long or do multiple unrelated tasks.
8. Whether notebook execution order currently matters in unsafe ways.

Produce this audit first in a markdown file:
`docs/notebook_refactor_audit.md`

Do not start large refactors before generating this audit.

---

## Implementation tasks

After the audit, execute the work in this order:

### Phase 1 — Safety and structure
- Create or improve a persistent path configuration system.
- Introduce a central config file for paths and runtime options.
- Ensure Google Colab-compatible path handling.
- Add clear output directories for checkpoints, metrics, and results.

### Phase 2 — Extract code from notebooks
- Move reusable logic into `src/`.
- Create functions for repeated operations such as:
  - loading data,
  - preprocessing,
  - creating dataloaders,
  - saving/loading checkpoints,
  - training one epoch,
  - validation/evaluation,
  - loading model for inference.

### Phase 3 — Checkpointing and resume
- Implement robust checkpoint saving/loading.
- Save:
  - model state,
  - optimizer state,
  - scheduler state if used,
  - epoch or global step,
  - best metric if applicable,
  - training history if practical.
- Add resume logic so training can continue after Colab interruption.

### Phase 4 — Notebook rewrite
- Rewrite notebooks to call functions from `src/` instead of embedding everything inline.
- Add markdown headings and beginner-friendly explanations.
- Ensure each notebook is top-down executable after fresh kernel restart.

### Phase 5 — Documentation
- Create `docs/COLAB_WORKFLOW.md` explaining:
  - what each notebook does,
  - in which order to run them,
  - what happens if runtime disconnects,
  - how to resume training,
  - where checkpoints are stored,
  - how to run eval/inference after a restart.

---

## Coding rules

Follow these coding rules:

- Make minimal but high-value changes.
- Do not rename everything unnecessarily.
- Preserve existing behavior where possible.
- Prefer readable code over “smart” code.
- Use clear function names.
- Add docstrings for new helper functions.
- Add comments only where they clarify intent.
- Do not add unnecessary dependencies.
- Do not hardcode Colab-only paths unless wrapped in config.
- Do not assume the environment is always Colab; support local Jupyter where possible.

---

## Validation requirements

Your changes are not complete unless you verify these conditions:

1. Training can start from scratch.
2. Training can resume from a saved checkpoint after interruption.
3. Evaluation can run in a fresh session after reloading artifacts/checkpoints.
4. Inference can run in a fresh session after reloading artifacts/checkpoints.
5. No later notebook requires hidden variables from earlier notebooks.
6. The new notebooks are shorter and easier to understand than the original.
7. All persistent outputs are written to explicit directories.
8. Documentation matches the actual code.

---

## Expected deliverables

Produce these deliverables:

- `docs/notebook_refactor_audit.md`
- refactored notebooks under `notebooks/`
- extracted modules under `src/`
- robust checkpoint helper(s)
- `docs/COLAB_WORKFLOW.md`

If appropriate, also add:
- a small smoke test script,
- a sample config file,
- a helper function to detect Colab vs local environment.

---

## Important constraints

Do not:
- leave the workflow dependent on execution order accidents,
- keep giant 200+ line notebook cells when extraction is possible,
- save only final weights if resume training requires more state,
- create documentation disconnected from the actual code,
- introduce breaking changes without explaining them.

Do:
- think like a repo maintainer,
- optimize for reliability under free Colab constraints,
- make the workflow usable for a beginner,
- produce explicit file paths and explicit save/load steps.

---

## Output format

Work in small, reviewable commits or patches.

When reporting progress, use this structure:

1. Audit findings
2. Proposed restructuring
3. Files created/modified
4. Risks or assumptions
5. Next recommended step

Do not give vague advice.
Inspect the codebase, make concrete changes, and explain only what is necessary.