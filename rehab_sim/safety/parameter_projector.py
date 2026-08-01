"""Independent parameter projection for RL actions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rehab_sim.controllers import AdmittanceParameters
from rehab_sim.safety.config import SafetyConfiguration

FloatArray = NDArray[np.float64]
PARAMETER_COUNT = 5
ACTION_COUNT = 4


def parameter_vector(parameters: AdmittanceParameters) -> FloatArray:
    """Return ``[Dx,Dy,Dtheta,Ka,lambda_v]`` from controller parameters."""

    return np.asarray(
        [*parameters.damping, parameters.assist_gain, parameters.velocity_scale],
        dtype=np.float64,
    )


def apply_parameter_vector(
    parameters: AdmittanceParameters, values: ArrayLike
) -> AdmittanceParameters:
    """Create controller parameters from a projected five-element vector."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (PARAMETER_COUNT,) or not np.all(np.isfinite(vector)):
        raise ValueError("projected parameters must be finite with shape (5,)")
    return replace(
        parameters,
        damping=vector[:3].copy(),
        assist_gain=float(vector[3]),
        velocity_scale=float(vector[4]),
    )


@dataclass(frozen=True)
class ProjectionResult:
    """Audit information and safe result of one action projection."""

    parameters: FloatArray
    action: FloatArray
    raw_action: FloatArray
    action_clipped: bool
    rate_limited: bool
    boundary_projected: bool
    fallback: bool
    reason: str | None


class SafeParameterProjector:
    """Project normalized policy increments into the configured safe set."""

    def __init__(self, configuration: SafetyConfiguration) -> None:
        self.configuration = configuration
        self._action_scales = np.asarray(
            [
                configuration.action_scales[0],
                configuration.action_scales[2],
                configuration.action_scales[3],
                configuration.action_scales[4],
            ],
            dtype=np.float64,
        )
        if np.any(self._action_scales <= 0.0):
            raise ValueError("action scales must be positive")

    def _fallback_result(self, raw_action: ArrayLike, reason: str) -> ProjectionResult:
        try:
            raw = np.asarray(raw_action, dtype=np.float64)
        except (TypeError, ValueError):
            raw = np.zeros(ACTION_COUNT, dtype=np.float64)
        if raw.shape != (ACTION_COUNT,):
            raw = np.zeros(ACTION_COUNT, dtype=np.float64)
        return ProjectionResult(
            parameters=self.configuration.fallback_parameters.copy(),
            action=np.zeros(ACTION_COUNT, dtype=np.float64),
            raw_action=raw.copy(),
            action_clipped=False,
            rate_limited=False,
            boundary_projected=False,
            fallback=True,
            reason=reason,
        )

    def fallback(self, raw_action: ArrayLike, reason: str) -> ProjectionResult:
        """Return the configured fallback parameters for a safety event."""

        return self._fallback_result(raw_action, reason)

    def project(self, raw_action: ArrayLike, current_parameters: ArrayLike) -> ProjectionResult:
        """Apply finite checks, action clipping, rate limits and bounds.

        The action has shape ``[dDxy,dDtheta,dKa,dlambda_v]`` and values in
        ``[-1,1]``. Parameters are ordered as ``[Dx,Dy,Dtheta,Ka,lambda_v]``.
        """

        try:
            raw = np.asarray(raw_action, dtype=np.float64)
            current = np.asarray(current_parameters, dtype=np.float64)
        except (TypeError, ValueError):
            return self._fallback_result(raw_action, "invalid_parameter_or_action_type")
        if raw.shape != (ACTION_COUNT,):
            return self._fallback_result(raw, "invalid_action_shape")
        if current.shape != (PARAMETER_COUNT,):
            return self._fallback_result(raw, "invalid_parameter_shape")
        if not np.all(np.isfinite(raw)):
            return self._fallback_result(raw, "non_finite_action")
        if not np.all(np.isfinite(current)):
            return self._fallback_result(raw, "non_finite_parameters")

        clipped_action = np.clip(raw, -1.0, 1.0)
        action_was_clipped = not np.array_equal(clipped_action, raw)
        requested_delta = np.asarray(
            [
                clipped_action[0] * self._action_scales[0],
                clipped_action[0] * self._action_scales[0],
                clipped_action[1] * self._action_scales[1],
                clipped_action[2] * self._action_scales[2],
                clipped_action[3] * self._action_scales[3],
            ],
            dtype=np.float64,
        )
        rate_limited_delta = np.clip(
            requested_delta,
            -self.configuration.parameter_rate_limits,
            self.configuration.parameter_rate_limits,
        )
        rate_limited = not np.array_equal(rate_limited_delta, requested_delta)
        candidate = current + rate_limited_delta
        projected = np.clip(
            candidate,
            self.configuration.parameter_lower,
            self.configuration.parameter_upper,
        )
        boundary_projected = not np.array_equal(projected, candidate)
        if not np.all(np.isfinite(projected)) or np.any(projected[:3] <= 0.0):
            return self._fallback_result(raw, "stability_rule_failed")
        safe_action = np.asarray(
            [
                (projected[0] - current[0]) / self._action_scales[0],
                (projected[2] - current[2]) / self._action_scales[1],
                (projected[3] - current[3]) / self._action_scales[2],
                (projected[4] - current[4]) / self._action_scales[3],
            ],
            dtype=np.float64,
        )
        return ProjectionResult(
            parameters=projected.copy(),
            action=np.clip(safe_action, -1.0, 1.0),
            raw_action=raw.copy(),
            action_clipped=action_was_clipped,
            rate_limited=rate_limited,
            boundary_projected=boundary_projected,
            fallback=False,
            reason=None,
        )
