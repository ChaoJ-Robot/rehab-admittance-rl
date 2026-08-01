"""Typed simulation safety configuration for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rehab_sim.controllers import AdmittanceParameters

FloatArray = NDArray[np.float64]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"safety config must contain a {name} mapping")
    return value


def _finite_vector(value: Any, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _parameter_range(section: dict[str, Any], name: str) -> tuple[float, float]:
    values = _finite_vector(section.get(name), 2, f"parameter_bounds.{name}")
    if values[0] > values[1]:
        raise ValueError(f"parameter_bounds.{name} lower bound exceeds upper bound")
    return float(values[0]), float(values[1])


@dataclass(frozen=True)
class SafetyConfiguration:
    """Validated safety limits and safe parameter set for simulation."""

    enabled: bool
    mode: str
    parameter_lower: FloatArray
    parameter_upper: FloatArray
    action_scales: FloatArray
    parameter_rate_limits: FloatArray
    fallback_parameters: FloatArray
    interaction_force_limit: float
    interaction_torque_limit: float
    task_speed_limit: float
    task_acceleration_limit: float
    policy_timeout_s: float
    force_warning_ratio: float
    hardware_validation_required: bool


def load_safety_configuration(
    admittance_config: dict[str, Any], safety_config: dict[str, Any]
) -> SafetyConfiguration:
    """Load Phase 6 safety limits from the two project YAML mappings."""

    safety = _mapping(safety_config.get("safety"), "safety")
    bounds = _mapping(safety.get("parameter_bounds"), "safety.parameter_bounds")
    rate_limits = _mapping(safety.get("parameter_rate_limits"), "safety.parameter_rate_limits")
    fallback = _mapping(safety.get("fallback_parameters"), "safety.fallback_parameters")
    adaptive = _mapping(admittance_config.get("adaptive_update"), "admittance.adaptive_update")
    action_scales = _mapping(adaptive.get("action_scales"), "adaptive_update.action_scales")

    bounds_order = (
        "damping_x",
        "damping_y",
        "damping_theta",
        "assist_gain",
        "velocity_scale",
    )
    ranges = [_parameter_range(bounds, name) for name in bounds_order]
    lower = np.asarray([item[0] for item in ranges], dtype=np.float64)
    upper = np.asarray([item[1] for item in ranges], dtype=np.float64)
    rate = _finite_vector(
        [
            rate_limits.get("damping_xy"),
            rate_limits.get("damping_xy"),
            rate_limits.get("damping_theta"),
            rate_limits.get("assist_gain"),
            rate_limits.get("velocity_scale"),
        ],
        5,
        "parameter_rate_limits",
    )
    scales = _finite_vector(
        [
            action_scales.get("damping_xy"),
            action_scales.get("damping_xy"),
            action_scales.get("damping_theta"),
            action_scales.get("assist_gain"),
            action_scales.get("velocity_scale"),
        ],
        5,
        "adaptive_update.action_scales",
    )
    if np.any(rate <= 0.0) or np.any(scales <= 0.0):
        raise ValueError("parameter rate limits and action scales must be positive")
    fallback_parameters = _finite_vector(
        [
            *_finite_vector(fallback.get("damping"), 3, "fallback_parameters.damping"),
            fallback.get("assist_gain"),
            fallback.get("velocity_scale"),
        ],
        5,
        "fallback_parameters",
    )
    if np.any(fallback_parameters < lower) or np.any(fallback_parameters > upper):
        raise ValueError("fallback_parameters must lie within parameter_bounds")
    # Reuse the baseline parser to ensure the fixed controller remains a valid
    # parameter source, without allowing the supervisor to modify its mass,
    # stiffness or low-level control fields.
    AdmittanceParameters.from_config(admittance_config)
    positive_limits = (
        "interaction_force_limit",
        "interaction_torque_limit",
        "task_speed_limit",
        "task_acceleration_limit",
    )
    limits = {name: float(safety[name]) for name in positive_limits}
    if any(value <= 0.0 or not np.isfinite(value) for value in limits.values()):
        raise ValueError("physical safety limits must be positive and finite")
    timeout_s = float(safety["policy_timeout_ms"]) / 1000.0
    warning_ratio = float(safety["force_warning_ratio"])
    if timeout_s <= 0.0 or not np.isfinite(timeout_s):
        raise ValueError("policy_timeout_ms must be positive and finite")
    if not 0.0 < warning_ratio <= 1.0:
        raise ValueError("force_warning_ratio must be in (0, 1]")
    return SafetyConfiguration(
        enabled=bool(safety.get("enabled")),
        mode=str(safety.get("mode")),
        parameter_lower=lower,
        parameter_upper=upper,
        action_scales=scales,
        parameter_rate_limits=rate,
        fallback_parameters=fallback_parameters,
        interaction_force_limit=limits["interaction_force_limit"],
        interaction_torque_limit=limits["interaction_torque_limit"],
        task_speed_limit=limits["task_speed_limit"],
        task_acceleration_limit=limits["task_acceleration_limit"],
        policy_timeout_s=timeout_s,
        force_warning_ratio=warning_ratio,
        hardware_validation_required=bool(safety.get("hardware_validation_required")),
    )
