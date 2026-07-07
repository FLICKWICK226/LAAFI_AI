"""Central path configuration for LAAFI_AI.

Resolves project root and output directories for both Google Colab
(with Google Drive persistence) and local Jupyter / script execution.
All persistent artifacts (checkpoints, metrics, figures, reports,
predictions) are written under explicit sub-directories so that
nothing depends on implicit runtime state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from laafi_ai.config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Environment detection
# ------------------------------------------------------------------


def is_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    try:
        import google.colab  # type: ignore[import-untyped]  # noqa: F401
        return True
    except ImportError:
        return False


def default_project_root() -> Path:
    """Best-effort guess for the project root directory.

    * Colab ➜ ``/content/drive/MyDrive/LAAFI_AI`` (common convention).
    * Local ➜ current working directory.
    """
    if is_colab():
        return Path("/content/drive/MyDrive/LAAFI_AI")
    return Path.cwd()


# ------------------------------------------------------------------
# ProjectPaths dataclass
# ------------------------------------------------------------------


@dataclass(slots=True)
class ProjectPaths:
    """Resolved, absolute paths for all persistent artifact directories.

    All directories are created eagerly when the instance is constructed
    via :func:`resolve_project_paths`.
    """

    project_root: Path
    output_dir: Path
    checkpoint_dir: Path
    metrics_dir: Path
    figures_dir: Path
    predictions_dir: Path
    reports_dir: Path


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def resolve_project_paths(
    config: ExperimentConfig,
    project_root: Path | None = None,
) -> ProjectPaths:
    """Build a :class:`ProjectPaths` from *config* and create every directory.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration; ``config.output_dir`` is used as the
        base for all artifact sub-directories.
    project_root : Path | None
        Explicit project root.  When *None*, :func:`default_project_root`
        is used.

    Returns
    -------
    ProjectPaths
        Fully resolved paths with all directories created on disk.

    Notes
    -----
    If ``config.output_dir`` is an **absolute** path it is used as-is;
    otherwise it is resolved relative to *project_root*.
    """
    if project_root is None:
        project_root = default_project_root()
    project_root = Path(project_root)

    output_path = Path(config.output_dir)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    checkpoint_dir = output_path / "checkpoints"
    metrics_dir = output_path / "metrics"
    figures_dir = output_path / "figures"
    predictions_dir = output_path / "predictions"
    reports_dir = output_path / "reports"

    # Create every directory eagerly so later code can write without checks.
    for directory in (output_path, checkpoint_dir, metrics_dir, figures_dir,
                      predictions_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(
        project_root=project_root,
        output_dir=output_path,
        checkpoint_dir=checkpoint_dir,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        predictions_dir=predictions_dir,
        reports_dir=reports_dir,
    )

    LOGGER.info("Project root: %s", paths.project_root)
    LOGGER.info("Output directory: %s", paths.output_dir)
    return paths
