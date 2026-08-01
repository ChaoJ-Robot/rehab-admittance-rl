"""Typed configuration for Phase 5 SAC experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rehab_sim.config import load_yaml


@dataclass(frozen=True)
class SACExperimentConfig:
    """Validated simulation configuration for one family of SAC runs."""

    total_timesteps: int
    random_seeds: tuple[int, ...]
    checkpoint_frequency: int
    evaluation_frequency: int
    evaluation_episodes: int
    task_name: str
    patient_profile: str
    n_envs: int
    normalization_enabled: bool
    normalize_observation: bool
    normalize_reward: bool
    clip_observation: float
    clip_reward: float
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    train_frequency: int
    gradient_steps: int
    ent_coef: str | float
    policy_net_arch: tuple[int, ...]
    device: str


def _mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"rl_sac.yaml must contain a {name} mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def load_sac_config(path: str | Path) -> SACExperimentConfig:
    """Load and validate a Phase 5 SAC configuration YAML file."""

    config = load_yaml(path)
    training = _mapping(config, "training")
    environment = _mapping(config, "environment")
    normalization = _mapping(config, "normalization")
    sac = _mapping(config, "sac")

    raw_seeds = training.get("random_seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("training.random_seeds must be a non-empty list")
    seeds = tuple(int(seed) for seed in raw_seeds)
    raw_arch = sac.get("policy_net_arch")
    if not isinstance(raw_arch, list) or not raw_arch:
        raise ValueError("sac.policy_net_arch must be a non-empty list")
    policy_net_arch = tuple(_positive_int(width, "policy_net_arch") for width in raw_arch)
    ent_coef = sac.get("ent_coef")
    if not isinstance(ent_coef, (str, int, float)):
        raise ValueError("sac.ent_coef must be 'auto' or a number")
    if isinstance(ent_coef, str) and ent_coef != "auto":
        raise ValueError("sac.ent_coef string value must be 'auto'")
    tau = float(sac["tau"])
    gamma = float(sac["gamma"])
    if tau > 1.0:
        raise ValueError("sac.tau must be in (0, 1]")
    if gamma <= 0.0 or gamma > 1.0:
        raise ValueError("sac.gamma must be in (0, 1]")

    return SACExperimentConfig(
        total_timesteps=_positive_int(training.get("total_timesteps"), "total_timesteps"),
        random_seeds=seeds,
        checkpoint_frequency=_positive_int(
            training.get("checkpoint_frequency"), "checkpoint_frequency"
        ),
        evaluation_frequency=_positive_int(
            training.get("evaluation_frequency"), "evaluation_frequency"
        ),
        evaluation_episodes=_positive_int(
            training.get("evaluation_episodes"), "evaluation_episodes"
        ),
        task_name=str(environment.get("task")),
        patient_profile=str(environment.get("patient_profile")),
        n_envs=_positive_int(environment.get("n_envs"), "environment.n_envs"),
        normalization_enabled=bool(normalization.get("enabled", True)),
        normalize_observation=bool(normalization.get("normalize_observation")),
        normalize_reward=bool(normalization.get("normalize_reward")),
        clip_observation=_positive_float(
            normalization.get("clip_observation"), "normalization.clip_observation"
        ),
        clip_reward=_positive_float(normalization.get("clip_reward"), "normalization.clip_reward"),
        learning_rate=_positive_float(sac.get("learning_rate"), "sac.learning_rate"),
        buffer_size=_positive_int(sac.get("buffer_size"), "sac.buffer_size"),
        learning_starts=_nonnegative_int(sac["learning_starts"], "sac.learning_starts"),
        batch_size=_positive_int(sac.get("batch_size"), "sac.batch_size"),
        tau=_positive_float(tau, "sac.tau"),
        gamma=gamma,
        train_frequency=_positive_int(sac.get("train_frequency"), "sac.train_frequency"),
        gradient_steps=_nonnegative_int(sac["gradient_steps"], "sac.gradient_steps"),
        ent_coef=ent_coef,
        policy_net_arch=policy_net_arch,
        device=str(sac.get("device")),
    )
