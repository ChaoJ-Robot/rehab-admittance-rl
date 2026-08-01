"""Stable-Baselines3 vector environment construction for Phase 5."""

from __future__ import annotations

from typing import TypeAlias

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from rehab_sim.envs import CircleTrackingEnv, Figure8TrackingEnv, PointReachEnv
from rehab_sim.rl.config import SACExperimentConfig

EnvironmentClass: TypeAlias = type[PointReachEnv | CircleTrackingEnv | Figure8TrackingEnv]


def environment_class(task_name: str) -> EnvironmentClass:
    """Return the Phase 4 environment class for a configured task name."""

    classes: dict[str, EnvironmentClass] = {
        "point_to_point": PointReachEnv,
        "circle_tracking": CircleTrackingEnv,
        "figure8_tracking": Figure8TrackingEnv,
    }
    try:
        return classes[task_name]
    except KeyError as error:
        raise ValueError(f"unknown Phase 5 task: {task_name}") from error


def make_normalized_env(
    config: SACExperimentConfig,
    seed: int,
    n_envs: int,
    training: bool,
) -> VecNormalize:
    """Create a vectorized Phase 4 environment with configured normalization.

    The environment action remains the four-dimensional normalized parameter
    increment defined in Phase 4. ``VecNormalize`` only transforms observations
    and rewards presented to SAC; it does not modify actions or controller
    safety limits.
    """

    base_env: VecEnv = make_vec_env(
        environment_class(config.task_name),
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"patient_profile": config.patient_profile},
    )
    return VecNormalize(
        base_env,
        training=training,
        norm_obs=config.normalization_enabled and config.normalize_observation,
        norm_reward=config.normalization_enabled and config.normalize_reward if training else False,
        clip_obs=config.clip_observation,
        clip_reward=config.clip_reward,
    )
