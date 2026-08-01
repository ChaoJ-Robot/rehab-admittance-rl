"""Configurable reward decomposition for rehabilitation tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class RewardWeights:
    """Weights from the YAML reward configuration."""

    progress: float
    normalized_tracking_error: float
    excessive_force_penalty: float
    motion_jerk_penalty: float
    robot_assistance_energy: float
    parameter_change: float
    positive_human_power: float
    task_success: float
    unsafe_termination: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> RewardWeights:
        """Load reward weights from the ``reward`` section."""

        raw = config.get("reward")
        if not isinstance(raw, Mapping):
            raise ValueError("RL config must contain a reward mapping")
        names = (
            "progress",
            "normalized_tracking_error",
            "excessive_force_penalty",
            "motion_jerk_penalty",
            "robot_assistance_energy",
            "parameter_change",
            "positive_human_power",
            "task_success",
            "unsafe_termination",
        )
        try:
            return cls(**{name: float(raw[name]) for name in names})
        except KeyError as error:
            raise ValueError(f"missing reward weight: {error.args[0]}") from error


@dataclass(frozen=True)
class RewardComponents:
    """Individual reward terms and their weighted sum."""

    progress: float
    normalized_tracking_error: float
    excessive_force_penalty: float
    motion_jerk_penalty: float
    robot_assistance_energy: float
    parameter_change: float
    positive_human_power: float
    task_success: float
    unsafe_termination: float

    def weighted_total(self, weights: RewardWeights) -> float:
        """Return the scalar reward using configured positive weights."""

        return (
            weights.progress * self.progress
            - weights.normalized_tracking_error * self.normalized_tracking_error
            - weights.excessive_force_penalty * self.excessive_force_penalty
            - weights.motion_jerk_penalty * self.motion_jerk_penalty
            - weights.robot_assistance_energy * self.robot_assistance_energy
            - weights.parameter_change * self.parameter_change
            + weights.positive_human_power * self.positive_human_power
            + weights.task_success * self.task_success
            - weights.unsafe_termination * self.unsafe_termination
        )

    def as_dict(self) -> dict[str, float]:
        """Return terms for Gymnasium ``info`` and experiment logging."""

        return {
            "progress": self.progress,
            "normalized_tracking_error": self.normalized_tracking_error,
            "excessive_force_penalty": self.excessive_force_penalty,
            "motion_jerk_penalty": self.motion_jerk_penalty,
            "robot_assistance_energy": self.robot_assistance_energy,
            "parameter_change": self.parameter_change,
            "positive_human_power": self.positive_human_power,
            "task_success": self.task_success,
            "unsafe_termination": self.unsafe_termination,
        }


def norm3(value: ArrayLike) -> float:
    """Return the Euclidean norm of a three-component task vector."""

    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))
