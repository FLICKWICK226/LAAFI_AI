from pathlib import Path

from laafi_ai.config import ExperimentConfig
from laafi_ai.paths import ProjectPaths, resolve_project_paths


def test_resolve_project_paths_creates_expected_artifact_directories(tmp_path: Path) -> None:
    config = ExperimentConfig(output_dir="outputs_test")

    paths = resolve_project_paths(config, project_root=tmp_path)

    assert isinstance(paths, ProjectPaths)
    assert paths.project_root == tmp_path
    assert paths.output_dir == tmp_path / "outputs_test"
    assert paths.checkpoint_dir == tmp_path / "outputs_test" / "checkpoints"
    assert paths.metrics_dir == tmp_path / "outputs_test" / "metrics"
    assert paths.figures_dir == tmp_path / "outputs_test" / "figures"
    assert paths.predictions_dir == tmp_path / "outputs_test" / "predictions"
    assert paths.reports_dir == tmp_path / "outputs_test" / "reports"

    for directory in (
        paths.output_dir,
        paths.checkpoint_dir,
        paths.metrics_dir,
        paths.figures_dir,
        paths.predictions_dir,
        paths.reports_dir,
    ):
        assert directory.exists()


def test_project_paths_accepts_absolute_output_dir(tmp_path: Path) -> None:
    absolute_output = tmp_path / "drive" / "laafi_outputs"
    config = ExperimentConfig(output_dir=str(absolute_output))

    paths = resolve_project_paths(config, project_root=tmp_path / "repo")

    assert paths.output_dir == absolute_output
    assert paths.checkpoint_dir == absolute_output / "checkpoints"
