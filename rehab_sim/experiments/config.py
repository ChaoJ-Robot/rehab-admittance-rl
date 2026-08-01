"""Typed configuration for Phase 10 comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rehab_sim.config import load_yaml

VALID_METHODS: tuple[str, ...] = (
    "fixed_admittance",
    "rule_adaptive",
    "fuzzy_control",
    "sac",
    "ppo",
)
VALID_PATIENTS: tuple[str, ...] = ("mild", "moderate", "severe")
VALID_TASKS: tuple[str, ...] = ("point_to_point", "circle_tracking", "figure8_tracking")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"phase10 config must contain a {name} mapping")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


@dataclass(frozen=True)
class Phase10Config:
    """Validated experiment matrix and deterministic policy parameters."""

    task_name: str
    patient_profiles: tuple[str, ...]
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    episodes_per_condition: int
    output_dir: Path
    rl_config: Path
    training_patient_profile: str
    training_timesteps: int
    device: str
    plot_dpi: int
    video_fps: int
    video_max_frames: int
    policy_parameters: dict[str, dict[str, float]]


def load_phase10_config(path: str | Path) -> Phase10Config:
    """Load and validate the Phase 10 YAML experiment specification."""

    config_path = Path(path)
    raw = load_yaml(config_path)
    experiment = _mapping(raw.get("experiment"), "experiment")
    training = _mapping(raw.get("training"), "training")
    plot = _mapping(raw.get("plot"), "plot")
    video = _mapping(raw.get("video"), "video")
    policies = _mapping(raw.get("policies"), "policies")

    task_name = str(experiment.get("task"))
    if task_name not in VALID_TASKS:
        raise ValueError(f"unknown Phase 10 task: {task_name}")
    raw_patients = experiment.get("patient_profiles")
    if not isinstance(raw_patients, list) or not raw_patients:
        raise ValueError("experiment.patient_profiles must be a non-empty list")
    patients = tuple(str(item) for item in raw_patients)
    if any(item not in VALID_PATIENTS for item in patients):
        raise ValueError("experiment.patient_profiles contains an unknown profile")
    raw_methods = experiment.get("methods")
    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError("experiment.methods must be a non-empty list")
    methods = tuple(str(item) for item in raw_methods)
    if len(set(methods)) != len(methods) or any(item not in VALID_METHODS for item in methods):
        raise ValueError("experiment.methods contains an unknown or repeated method")
    raw_seeds = experiment.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("experiment.seeds must be a non-empty list")
    seeds = tuple(int(item) for item in raw_seeds)
    training_patient = str(training.get("patient_profile"))
    if training_patient not in VALID_PATIENTS:
        raise ValueError("training.patient_profile is unknown")

    policy_parameters: dict[str, dict[str, float]] = {}
    for method in ("rule_adaptive", "fuzzy_control"):
        section = _mapping(policies.get(method), f"policies.{method}")
        policy_parameters[method] = {name: float(value) for name, value in section.items()}

    return Phase10Config(
        task_name=task_name,
        patient_profiles=patients,
        methods=methods,
        seeds=seeds,
        episodes_per_condition=_positive_int(
            experiment.get("episodes_per_condition"), "episodes_per_condition"
        ),
        output_dir=Path(str(experiment.get("output_dir"))),
        rl_config=Path(str(experiment.get("rl_config"))),
        training_patient_profile=training_patient,
        training_timesteps=_nonnegative_int(training.get("timesteps"), "training.timesteps"),
        device=str(training.get("device", "cpu")),
        plot_dpi=_positive_int(plot.get("dpi"), "plot.dpi"),
        video_fps=_positive_int(video.get("fps"), "video.fps"),
        video_max_frames=_positive_int(video.get("max_frames"), "video.max_frames"),
        policy_parameters=policy_parameters,
    )
