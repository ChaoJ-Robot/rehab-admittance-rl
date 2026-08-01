"""YAML configuration loading for the rehabilitation robot project.

Configuration values are returned as nested dictionaries so later phases can
add schema-specific models without changing the file-loading API. This module
does not apply robot or controller defaults.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILENAMES: tuple[str, ...] = (
    "robot.yaml",
    "admittance.yaml",
    "safety.yaml",
    "patient_profiles.yaml",
    "tasks.yaml",
    "rl_sac.yaml",
)


class ConfigError(ValueError):
    """Raised when a YAML configuration cannot be loaded as a mapping."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping from *path* without inventing missing values.

    Args:
        path: YAML file path.

    Returns:
        A mapping containing the parsed YAML data. An empty YAML document
        returns an empty mapping.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ConfigError: If the document root is not a mapping.
        yaml.YAMLError: If the document is invalid YAML.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    return data


def load_config_bundle(
    config_dir: str | Path,
    filenames: Iterable[str] = CONFIG_FILENAMES,
) -> dict[str, dict[str, Any]]:
    """Load the named YAML files from a configuration directory.

    The returned keys are file stems, for example ``robot`` for
    ``robot.yaml``. Missing files are allowed only when the caller does not
    request them; the default Phase 0 bundle requires all six templates.
    """

    root = Path(config_dir)
    bundle: dict[str, dict[str, Any]] = {}
    for filename in filenames:
        path = root / filename
        bundle[path.stem] = load_yaml(path)
    return bundle
