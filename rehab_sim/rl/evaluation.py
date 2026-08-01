"""Evaluation metrics for trained and random Phase 5 policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from rehab_sim.rl.vector_env import environment_class


@dataclass
class _EpisodeAccumulator:
    """Metrics collected for one finite-horizon episode."""

    reward: float = 0.0
    length: int = 0
    success: bool = False
    unsafe: bool = False
    max_force: float = 0.0
    force_sum: float = 0.0
    parameter_changes: list[float] = field(default_factory=list)
    parameter_deltas: list[np.ndarray] = field(default_factory=list)
    previous_parameters: np.ndarray | None = None

    def add_step(self, reward: float, info: dict[str, Any]) -> None:
        self.reward += float(reward)
        self.length += 1
        force_norm = float(info.get("interaction_force_norm", 0.0))
        self.max_force = max(self.max_force, force_norm)
        self.force_sum += force_norm
        parameters_value = info.get("admittance_parameters")
        if parameters_value is not None:
            parameters = np.asarray(parameters_value, dtype=np.float64)
            if self.previous_parameters is not None:
                delta = parameters - self.previous_parameters
                self.parameter_deltas.append(delta)
                self.parameter_changes.append(float(np.linalg.norm(delta)))
            self.previous_parameters = parameters
        self.success = bool(info.get("is_success", self.success))
        self.unsafe = bool(info.get("unsafe_reason") is not None or self.unsafe)

    def finish(self, info: dict[str, Any]) -> None:
        self.success = bool(info.get("is_success", self.success))
        self.unsafe = bool(info.get("unsafe_reason") is not None or self.unsafe)

    def as_dict(self) -> dict[str, float | int | bool]:
        sign_changes = 0
        for previous, current in zip(
            self.parameter_deltas, self.parameter_deltas[1:], strict=False
        ):
            previous_sign = np.sign(previous)
            current_sign = np.sign(current)
            sign_changes += int(
                np.any(
                    (previous_sign != 0.0) & (current_sign != 0.0) & (previous_sign != current_sign)
                )
            )
        transitions = max(1, len(self.parameter_deltas) - 1)
        return {
            "reward": self.reward,
            "length": self.length,
            "success": self.success,
            "unsafe": self.unsafe,
            "max_interaction_force": self.max_force,
            "mean_interaction_force": self.force_sum / max(1, self.length),
            "mean_parameter_change": float(np.mean(self.parameter_changes))
            if self.parameter_changes
            else 0.0,
            "parameter_oscillation_rate": sign_changes / transitions,
        }


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics required for Phase 5 acceptance."""

    episodes: int
    reward_mean: float
    reward_std: float
    success_rate: float
    unsafe_rate: float
    episode_length_mean: float
    max_interaction_force: float
    mean_interaction_force: float
    mean_parameter_change: float
    parameter_oscillation_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "episodes": self.episodes,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "success_rate": self.success_rate,
            "unsafe_rate": self.unsafe_rate,
            "episode_length_mean": self.episode_length_mean,
            "max_interaction_force": self.max_interaction_force,
            "mean_interaction_force": self.mean_interaction_force,
            "mean_parameter_change": self.mean_parameter_change,
            "parameter_oscillation_rate": self.parameter_oscillation_rate,
        }


def _aggregate(episodes: list[_EpisodeAccumulator]) -> EvaluationSummary:
    if not episodes:
        raise ValueError("at least one completed episode is required")
    values = [episode.as_dict() for episode in episodes]
    return EvaluationSummary(
        episodes=len(values),
        reward_mean=float(np.mean([float(value["reward"]) for value in values])),
        reward_std=float(np.std([float(value["reward"]) for value in values])),
        success_rate=float(np.mean([float(bool(value["success"])) for value in values])),
        unsafe_rate=float(np.mean([float(bool(value["unsafe"])) for value in values])),
        episode_length_mean=float(np.mean([float(value["length"]) for value in values])),
        max_interaction_force=float(max(float(value["max_interaction_force"]) for value in values)),
        mean_interaction_force=float(
            np.mean([float(value["mean_interaction_force"]) for value in values])
        ),
        mean_parameter_change=float(
            np.mean([float(value["mean_parameter_change"]) for value in values])
        ),
        parameter_oscillation_rate=float(
            np.mean([float(value["parameter_oscillation_rate"]) for value in values])
        ),
    )


def evaluate_random_policy(
    task_name: str,
    patient_profile: str,
    episodes: int,
    seed: int,
) -> EvaluationSummary:
    """Evaluate uniformly sampled bounded actions on the raw environment."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env_type = environment_class(task_name)
    environment = env_type(patient_profile=patient_profile, seed=seed)
    completed: list[_EpisodeAccumulator] = []
    try:
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            observation, _ = environment.reset(seed=episode_seed)
            _ = observation
            environment.action_space.seed(episode_seed)
            accumulator = _EpisodeAccumulator()
            while True:
                _, reward, terminated, truncated, info = environment.step(
                    environment.action_space.sample()
                )
                accumulator.add_step(float(reward), info)
                if terminated or truncated:
                    accumulator.finish(info)
                    completed.append(accumulator)
                    break
    finally:
        environment.close()
    return _aggregate(completed)


def evaluate_model(
    model: Any,
    environment: VecNormalize,
    episodes: int,
    deterministic: bool = True,
) -> EvaluationSummary:
    """Evaluate a loaded SB3 model on a one-environment VecNormalize wrapper."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if environment.num_envs != 1:
        raise ValueError("evaluate_model requires a one-environment VecNormalize wrapper")
    observation = environment.reset()
    completed: list[_EpisodeAccumulator] = []
    accumulator = _EpisodeAccumulator()
    while len(completed) < episodes:
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, rewards, dones, infos = environment.step(action)
        info = dict(infos[0])
        accumulator.add_step(float(rewards[0]), info)
        if bool(dones[0]):
            accumulator.finish(info)
            completed.append(accumulator)
            accumulator = _EpisodeAccumulator()
    return _aggregate(completed)
