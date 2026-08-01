from pathlib import Path

import pytest
from rehab_sim.config import ConfigError, load_config_bundle, load_yaml


def test_phase_zero_templates_load_as_mappings() -> None:
    config_dir = Path(__file__).parents[2] / "configs"
    bundle = load_config_bundle(config_dir)

    assert set(bundle) == {
        "robot",
        "admittance",
        "safety",
        "patient_profiles",
        "tasks",
        "rl_sac",
        "agent",
        "ros2",
        "phase10",
    }
    assert all(isinstance(value, dict) for value in bundle.values())


def test_load_yaml_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_yaml(config_path)


def test_empty_yaml_is_an_empty_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_yaml(config_path) == {}
