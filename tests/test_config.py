from laafi_ai.config import ExperimentConfig


def test_default_config_has_expected_dataset() -> None:
    config = ExperimentConfig()
    assert config.data.dataset_name == "1aurent/PatchCamelyon"
    assert config.model.architecture == "resnet50"
