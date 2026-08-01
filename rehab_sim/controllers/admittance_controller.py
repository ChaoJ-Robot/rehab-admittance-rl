"""Fixed-parameter task-space admittance controller.

The controller operates on task vectors ``X=[x,y,theta]`` in m, m, rad and
interaction wrench vectors ``F=[Fx,Fy,Tz]`` in N, N, N*m. It outputs a
bounded desired task pose and velocity; it never writes MuJoCo actuator
commands and contains no reinforcement-learning logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rehab_sim.robot.kinematics import WorkspaceBounds, _vector, wrap_angle

FloatArray = NDArray[np.float64]


def _triple(section: Mapping[str, Any], names: tuple[str, str, str], label: str) -> FloatArray:
    """Read an ordered x/y/theta triple from a YAML mapping."""

    try:
        value = [section[name] for name in names]
    except KeyError as error:
        raise ValueError(f"missing {label}.{error.args[0]}") from error
    return _vector(value, 3, label)


@dataclass(frozen=True)
class AdmittanceParameters:
    """Validated fixed parameters for a diagonal three-DOF admittance model."""

    sample_time_s: float
    mass: FloatArray
    damping: FloatArray
    stiffness: FloatArray
    assist_gain: float
    assist_damping: float
    velocity_limits: FloatArray
    acceleration_limits: FloatArray
    force_filter_time_constant_s: FloatArray
    force_deadzone: FloatArray
    velocity_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "mass",
            "damping",
            "stiffness",
            "velocity_limits",
            "acceleration_limits",
            "force_filter_time_constant_s",
            "force_deadzone",
        ):
            value = _vector(getattr(self, name), 3, name)
            object.__setattr__(self, name, value)
        if self.sample_time_s <= 0.0:
            raise ValueError("sample_time_s must be positive")
        for name in ("mass", "damping", "velocity_limits", "acceleration_limits"):
            if np.any(getattr(self, name) <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
        if np.any(self.stiffness < 0.0):
            raise ValueError("stiffness must be non-negative")
        if np.any(self.force_filter_time_constant_s < 0.0):
            raise ValueError("force filter time constants must be non-negative")
        if np.any(self.force_deadzone < 0.0):
            raise ValueError("force deadzones must be non-negative")
        if self.assist_gain < 0.0 or self.assist_damping < 0.0:
            raise ValueError("assist gains must be non-negative")
        if self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be positive")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AdmittanceParameters:
        """Load the fixed simulation baseline from ``admittance.yaml``."""

        section = config.get("admittance")
        processing = config.get("force_processing")
        if not isinstance(section, Mapping) or not isinstance(processing, Mapping):
            raise ValueError(
                "admittance config must contain admittance and force_processing mappings"
            )
        return cls(
            sample_time_s=float(section["sample_time_s"]),
            mass=_triple(section["mass"], ("x", "y", "theta"), "mass"),
            damping=_triple(section["damping"], ("x", "y", "theta"), "damping"),
            stiffness=_triple(section["stiffness"], ("x", "y", "theta"), "stiffness"),
            assist_gain=float(section["assist_gain"]),
            assist_damping=float(section["assist_damping"]),
            velocity_limits=_triple(
                section["velocity_limits"], ("x", "y", "theta"), "velocity_limits"
            ),
            acceleration_limits=_triple(
                section["acceleration_limits"], ("x", "y", "theta"), "acceleration_limits"
            ),
            force_filter_time_constant_s=_triple(
                processing["low_pass_time_constant_s"],
                ("x", "y", "theta"),
                "low_pass_time_constant_s",
            ),
            force_deadzone=_triple(
                processing["soft_deadzone"], ("x", "y", "theta"), "soft_deadzone"
            ),
            velocity_scale=float(section.get("velocity_scale", 1.0)),
        )


@dataclass(frozen=True)
class AdmittanceOutput:
    """One controller update result in task-space SI units."""

    desired_pose: FloatArray
    desired_velocity: FloatArray
    desired_acceleration: FloatArray
    filtered_wrench: FloatArray
    effective_wrench: FloatArray
    assist_force: FloatArray


class AdmittanceController:
    """Discrete-time diagonal admittance with bounded task-space output."""

    def __init__(
        self,
        parameters: AdmittanceParameters,
        workspace: WorkspaceBounds | None = None,
    ) -> None:
        self.parameters = parameters
        self.workspace = workspace
        self._initialized = False
        self._desired_pose = np.zeros(3, dtype=np.float64)
        self._desired_velocity = np.zeros(3, dtype=np.float64)
        self._filtered_wrench = np.zeros(3, dtype=np.float64)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        workspace: WorkspaceBounds | None = None,
    ) -> AdmittanceController:
        """Construct a controller from the YAML configuration mapping."""

        return cls(AdmittanceParameters.from_config(config), workspace=workspace)

    @property
    def desired_pose(self) -> FloatArray:
        """Current bounded desired task pose."""

        return self._desired_pose.copy()

    @property
    def desired_velocity(self) -> FloatArray:
        """Current bounded desired task velocity."""

        return self._desired_velocity.copy()

    @property
    def filtered_wrench(self) -> FloatArray:
        """Current low-pass-filtered wrench before deadzone processing."""

        return self._filtered_wrench.copy()

    def reset(self, pose: ArrayLike, velocity: ArrayLike | None = None) -> None:
        """Reset the dynamic state to a measured task pose and velocity."""

        self._desired_pose[:] = _vector(pose, 3, "pose")
        self._desired_velocity[:] = (
            np.zeros(3, dtype=np.float64) if velocity is None else _vector(velocity, 3, "velocity")
        )
        self._filtered_wrench[:] = 0.0
        self._initialized = True

    def update(
        self,
        measured_pose: ArrayLike,
        measured_velocity: ArrayLike,
        reference_pose: ArrayLike,
        reference_velocity: ArrayLike | None,
        interaction_wrench: ArrayLike,
    ) -> AdmittanceOutput:
        """Advance the admittance state by one sample.

        The measured pose initializes the command state on the first call. The
        acceleration equation is:

        ``M*ddX + D*dX + K*(X-Xr) = F_effective + F_assist``.
        """

        measured = _vector(measured_pose, 3, "measured_pose")
        _ = _vector(measured_velocity, 3, "measured_velocity")
        reference = _vector(reference_pose, 3, "reference_pose")
        reference_rate = (
            np.zeros(3, dtype=np.float64)
            if reference_velocity is None
            else _vector(reference_velocity, 3, "reference_velocity")
        )
        wrench = _vector(interaction_wrench, 3, "interaction_wrench")
        if not self._initialized:
            self.reset(measured)

        dt = self.parameters.sample_time_s
        tau = self.parameters.force_filter_time_constant_s
        alpha = np.ones(3, dtype=np.float64)
        positive_tau = tau > 0.0
        alpha[positive_tau] = dt / (tau[positive_tau] + dt)
        self._filtered_wrench += alpha * (wrench - self._filtered_wrench)
        effective_wrench = np.sign(self._filtered_wrench) * np.maximum(
            np.abs(self._filtered_wrench) - self.parameters.force_deadzone, 0.0
        )

        pose_error = reference - self._desired_pose
        pose_error[2] = wrap_angle(float(pose_error[2]))
        assist_force = self.parameters.assist_gain * pose_error + self.parameters.assist_damping * (
            reference_rate - self._desired_velocity
        )
        restoring_force = self.parameters.stiffness * (self._desired_pose - reference)
        acceleration = (
            effective_wrench
            + assist_force
            - self.parameters.damping * self._desired_velocity
            - restoring_force
        ) / self.parameters.mass
        acceleration = np.clip(
            acceleration,
            -self.parameters.acceleration_limits,
            self.parameters.acceleration_limits,
        )

        velocity = self._desired_velocity + acceleration * dt
        velocity = np.clip(
            velocity,
            -self.parameters.velocity_limits * self.parameters.velocity_scale,
            self.parameters.velocity_limits * self.parameters.velocity_scale,
        )
        candidate_pose = self._desired_pose + velocity * dt
        candidate_pose[2] = wrap_angle(float(candidate_pose[2]))
        if self.workspace is not None:
            clipped_pose = self.workspace.clip(candidate_pose)
            velocity = (clipped_pose - self._desired_pose) / dt
            candidate_pose = clipped_pose
            candidate_pose[2] = wrap_angle(float(candidate_pose[2]))

        self._desired_velocity[:] = velocity
        self._desired_pose[:] = candidate_pose
        return AdmittanceOutput(
            desired_pose=self._desired_pose.copy(),
            desired_velocity=self._desired_velocity.copy(),
            desired_acceleration=acceleration.copy(),
            filtered_wrench=self._filtered_wrench.copy(),
            effective_wrench=effective_wrench.copy(),
            assist_force=assist_force.copy(),
        )
