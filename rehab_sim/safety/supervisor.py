"""Independent safety supervisor for parameter adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rehab_sim.safety.config import SafetyConfiguration
from rehab_sim.safety.parameter_projector import (
    ProjectionResult,
    SafeParameterProjector,
)

FloatArray = NDArray[np.float64]


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array.copy()


@dataclass(frozen=True)
class SafetyObservation:
    """Safety signals in task-space units.

    ``interaction_wrench`` is ``[Fx,Fy,Tz]`` in N, N, N*m. Velocity and
    acceleration are ``[vx,vy,omega]`` in m/s, m/s, rad/s and corresponding
    per-second units. ``None`` acceleration means it is unavailable and is not
    checked by this software-only supervisor.
    """

    interaction_wrench: FloatArray
    task_velocity: FloatArray
    task_acceleration: FloatArray | None = None
    sensor_ok: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interaction_wrench",
            _vector(self.interaction_wrench, 3, "interaction_wrench"),
        )
        object.__setattr__(self, "task_velocity", _vector(self.task_velocity, 3, "task_velocity"))
        if self.task_acceleration is not None:
            object.__setattr__(
                self,
                "task_acceleration",
                _vector(self.task_acceleration, 3, "task_acceleration"),
            )


@dataclass(frozen=True)
class SafetyDecision:
    """Approved parameters or a fail-safe fallback decision."""

    parameters: FloatArray
    action: FloatArray
    raw_action: FloatArray
    approved: bool
    fallback: bool
    reasons: tuple[str, ...]
    force_norm: float
    torque_abs: float
    speed_norm: float
    acceleration_norm: float | None
    projection: ProjectionResult

    @property
    def reason(self) -> str | None:
        """Return a compact reason string for logs and telemetry."""

        return ";".join(self.reasons) if self.reasons else None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly decision data."""

        return {
            "approved": self.approved,
            "fallback": self.fallback,
            "reasons": list(self.reasons),
            "force_norm": self.force_norm,
            "torque_abs": self.torque_abs,
            "speed_norm": self.speed_norm,
            "acceleration_norm": self.acceleration_norm,
            "parameters": self.parameters.tolist(),
            "action": self.action.tolist(),
            "raw_action": self.raw_action.tolist(),
            "projection": {
                "action_clipped": self.projection.action_clipped,
                "rate_limited": self.projection.rate_limited,
                "boundary_projected": self.projection.boundary_projected,
                "fallback": self.projection.fallback,
            },
        }


class SafetySupervisor:
    """Fail-closed supervisor independent from any RL implementation."""

    def __init__(self, configuration: SafetyConfiguration) -> None:
        self.configuration = configuration
        self.projector = SafeParameterProjector(configuration)

    def _metrics(self, observation: SafetyObservation) -> tuple[float, float, float, float | None]:
        wrench = observation.interaction_wrench
        acceleration = observation.task_acceleration
        return (
            float(np.linalg.norm(wrench[:2])),
            float(abs(wrench[2])),
            float(np.linalg.norm(observation.task_velocity)),
            None if acceleration is None else float(np.linalg.norm(acceleration)),
        )

    def _state_reasons(self, observation: SafetyObservation) -> tuple[str, ...]:
        if not observation.sensor_ok:
            return ("sensor_disconnected",)
        arrays = [observation.interaction_wrench, observation.task_velocity]
        if observation.task_acceleration is not None:
            arrays.append(observation.task_acceleration)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            return ("non_finite_state",)
        force_norm, torque_abs, speed_norm, acceleration_norm = self._metrics(observation)
        reasons: list[str] = []
        if (
            force_norm
            >= self.configuration.interaction_force_limit * self.configuration.force_warning_ratio
        ):
            reasons.append("interaction_force_near_limit")
        if torque_abs > self.configuration.interaction_torque_limit:
            reasons.append("interaction_torque_limit")
        if speed_norm > self.configuration.task_speed_limit:
            reasons.append("task_speed_limit")
        if (
            acceleration_norm is not None
            and acceleration_norm > self.configuration.task_acceleration_limit
        ):
            reasons.append("task_acceleration_limit")
        return tuple(reasons)

    def fallback(
        self,
        current_parameters: ArrayLike,
        raw_action: ArrayLike,
        reasons: tuple[str, ...],
        observation: SafetyObservation | None = None,
    ) -> SafetyDecision:
        """Build a fail-safe decision without consulting an RL model."""

        projection = self.projector.fallback(raw_action, ";".join(reasons))
        if observation is None:
            force_norm = torque_abs = speed_norm = 0.0
            acceleration_norm = None
        else:
            force_norm, torque_abs, speed_norm, acceleration_norm = self._metrics(observation)
        return SafetyDecision(
            parameters=projection.parameters.copy(),
            action=projection.action.copy(),
            raw_action=projection.raw_action.copy(),
            approved=False,
            fallback=True,
            reasons=reasons,
            force_norm=force_norm,
            torque_abs=torque_abs,
            speed_norm=speed_norm,
            acceleration_norm=acceleration_norm,
            projection=projection,
        )

    def supervise(
        self,
        raw_action: ArrayLike,
        current_parameters: ArrayLike,
        observation: SafetyObservation,
        inference_elapsed_s: float = 0.0,
        model_loaded: bool = True,
    ) -> SafetyDecision:
        """Approve one policy action or immediately select fallback parameters."""

        reasons: list[str] = []
        if not self.configuration.enabled:
            reasons.append("safety_supervisor_disabled")
        if not model_loaded:
            reasons.append("model_not_loaded")
        try:
            inference_time_valid = bool(np.isfinite(inference_elapsed_s))
        except TypeError:
            inference_time_valid = False
        if not inference_time_valid or inference_elapsed_s > self.configuration.policy_timeout_s:
            reasons.append("policy_timeout")
        try:
            raw = np.asarray(raw_action, dtype=np.float64)
        except (TypeError, ValueError):
            raw = np.zeros(4, dtype=np.float64)
            reasons.append("invalid_action_type")
        if raw.shape != (4,) or not np.all(np.isfinite(raw)):
            reasons.append("non_finite_or_invalid_action")
        reasons.extend(self._state_reasons(observation))
        if reasons:
            return self.fallback(
                current_parameters, raw, tuple(dict.fromkeys(reasons)), observation
            )

        projection = self.projector.project(raw, current_parameters)
        if projection.fallback:
            return self.fallback(
                current_parameters,
                raw,
                (projection.reason or "projection_failed",),
                observation,
            )
        force_norm, torque_abs, speed_norm, acceleration_norm = self._metrics(observation)
        return SafetyDecision(
            parameters=projection.parameters.copy(),
            action=projection.action.copy(),
            raw_action=projection.raw_action.copy(),
            approved=True,
            fallback=False,
            reasons=(),
            force_norm=force_norm,
            torque_abs=torque_abs,
            speed_norm=speed_norm,
            acceleration_norm=acceleration_norm,
            projection=projection,
        )
