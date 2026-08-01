"""Small ROS 2 package configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def find_config_dir(config_dir: str = "") -> Path:
    """Resolve a project ``configs`` directory without inventing hardware data."""

    candidates = []
    if config_dir:
        candidates.append(Path(config_dir))
    env_dir = os.environ.get("REHAB_CONFIG_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path.cwd() / "configs",
            Path(__file__).resolve().parents[4] / "configs",
        ]
    )
    for candidate in candidates:
        if (candidate / "ros2.yaml").is_file():
            return candidate
    raise FileNotFoundError("could not find configs/ros2.yaml; set config_dir or REHAB_CONFIG_DIR")


def load_ros_config(config_dir: str = "") -> dict[str, Any]:
    """Load and validate the Phase 9 ROS 2 YAML mapping."""

    directory = find_config_dir(config_dir)
    with (directory / "ros2.yaml").open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("ros2"), dict):
        raise ValueError("ros2.yaml must contain a top-level ros2 mapping")
    return data["ros2"]
