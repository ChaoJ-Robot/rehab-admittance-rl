"""Hardware-independent robot I/O contracts for the ROS 2 boundary.

All pose and velocity vectors use task-space order ``[x, y, theta]`` in m,
m/s, and rad/s. Wrenches use ``[Fx, Fy, Tz]`` in N, N, and N*m. The adapter
publishes or accepts admittance parameters only; it never accepts motor
torques, currents, or joint commands.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
CommandSink = Callable[["AdmittanceCommand"], None]


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    """Validate and copy a finite vector with an explicit dimension."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


@dataclass(frozen=True)
class RobotState:
    """Normalized state shared by simulation and ROS 2 robot adapters."""

    timestamp_s: float
    joint_position: FloatArray
    joint_velocity: FloatArray
    end_effector_pose: FloatArray
    end_effector_velocity: FloatArray
    wrench: FloatArray
    sensor_ok: bool
    source: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        object.__setattr__(
            self, "joint_position", _vector(self.joint_position, 3, "joint_position")
        )
        object.__setattr__(
            self, "joint_velocity", _vector(self.joint_velocity, 3, "joint_velocity")
        )
        object.__setattr__(
            self,
            "end_effector_pose",
            _vector(self.end_effector_pose, 3, "end_effector_pose"),
        )
        object.__setattr__(
            self,
            "end_effector_velocity",
            _vector(self.end_effector_velocity, 3, "end_effector_velocity"),
        )
        object.__setattr__(self, "wrench", _vector(self.wrench, 3, "wrench"))


@dataclass(frozen=True)
class AdmittanceCommand:
    """A safe task-space parameter command, never a motor command.

    Parameters are ordered as ``D=[Dx,Dy,Dtheta]``, ``Ka`` and ``lambda_v``.
    ``lambda_v`` is a dimensionless velocity-limit scale. The source and
    fallback flags are retained for audit logs and downstream gating.
    """

    damping: FloatArray
    assist_gain: float
    velocity_scale: float
    source: str
    fallback: bool = False
    low_speed_test: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "damping", _vector(self.damping, 3, "damping"))
        if not np.isfinite(self.assist_gain) or self.assist_gain < 0.0:
            raise ValueError("assist_gain must be finite and non-negative")
        if not np.isfinite(self.velocity_scale) or self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be finite and positive")
        if not self.source:
            raise ValueError("source must not be empty")

    @classmethod
    def from_vector(
        cls,
        values: ArrayLike,
        *,
        source: str,
        fallback: bool = False,
        low_speed_test: bool = False,
    ) -> AdmittanceCommand:
        """Construct a command from ``[Dx,Dy,Dtheta,Ka,lambda_v]``."""

        vector = _vector(values, 5, "parameters")
        return cls(
            damping=vector[:3],
            assist_gain=float(vector[3]),
            velocity_scale=float(vector[4]),
            source=source,
            fallback=fallback,
            low_speed_test=low_speed_test,
        )

    def as_vector(self) -> FloatArray:
        """Return the five-element parameter vector."""

        return np.asarray([*self.damping, self.assist_gain, self.velocity_scale], dtype=np.float64)


class RobotAdapter(Protocol):
    """Unified read/parameter-write contract for simulation and hardware."""

    def read_state(self) -> RobotState:
        """Read one normalized state sample."""

    def write_admittance_parameters(self, command: AdmittanceCommand) -> None:
        """Send task-space admittance parameters to the downstream controller."""

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the downstream parameter consumer."""

    def reset(self) -> None:
        """Reset adapter state without touching motor-level interfaces."""


class SimulationRobotAdapter:
    """In-memory adapter exposing the same contract as the ROS 2 bridge."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._state = RobotState(
            timestamp_s=float(clock()),
            joint_position=np.zeros(3),
            joint_velocity=np.zeros(3),
            end_effector_pose=np.zeros(3),
            end_effector_velocity=np.zeros(3),
            wrench=np.zeros(3),
            sensor_ok=True,
            source="simulation",
        )
        self._last_command: AdmittanceCommand | None = None
        self._enabled = False

    @property
    def last_command(self) -> AdmittanceCommand | None:
        """Last parameter command accepted by the simulation adapter."""

        return self._last_command

    @property
    def enabled(self) -> bool:
        """Whether the downstream simulation controller is enabled."""

        return self._enabled

    def inject_state(
        self,
        *,
        joint_position: ArrayLike,
        joint_velocity: ArrayLike,
        end_effector_pose: ArrayLike,
        end_effector_velocity: ArrayLike,
        wrench: ArrayLike,
        sensor_ok: bool = True,
    ) -> None:
        """Inject a simulation sample for deterministic tests or playback."""

        self._state = RobotState(
            timestamp_s=float(self._clock()),
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            end_effector_pose=end_effector_pose,
            end_effector_velocity=end_effector_velocity,
            wrench=wrench,
            sensor_ok=sensor_ok,
            source="simulation",
        )

    def read_state(self) -> RobotState:
        return self._state

    def write_admittance_parameters(self, command: AdmittanceCommand) -> None:
        self._last_command = command

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def reset(self) -> None:
        self._last_command = None
        self._enabled = False


class Ros2RobotAdapter:
    """ROS-message-facing adapter with the same contract as simulation.

    A downstream driver receives only :class:`AdmittanceCommand`. The adapter
    intentionally has no method for torque/current publication. Invalid input
    marks the normalized state unavailable while retaining the last finite
    sample for diagnostics.
    """

    def __init__(
        self,
        command_sink: CommandSink | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._command_sink = command_sink
        self._joint_position = np.zeros(3, dtype=np.float64)
        self._joint_velocity = np.zeros(3, dtype=np.float64)
        self._pose = np.zeros(3, dtype=np.float64)
        self._velocity = np.zeros(3, dtype=np.float64)
        self._wrench = np.zeros(3, dtype=np.float64)
        self._joint_ok = False
        self._pose_ok = False
        self._wrench_ok = False
        self._last_timestamp_s = float(clock())
        self._last_command: AdmittanceCommand | None = None
        self._enabled = False

    @property
    def last_command(self) -> AdmittanceCommand | None:
        """Last parameter command received from the policy/safety chain."""

        return self._last_command

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update_joint_state(self, position: ArrayLike, velocity: ArrayLike) -> bool:
        """Update joint state in the configured planar three-joint order."""

        try:
            self._joint_position = _vector(position, 3, "joint_position")
            self._joint_velocity = _vector(velocity, 3, "joint_velocity")
        except (TypeError, ValueError):
            self._joint_ok = False
            return False
        self._joint_ok = True
        self._last_timestamp_s = float(self._clock())
        return True

    def update_end_effector(self, pose: ArrayLike, velocity: ArrayLike, valid: bool = True) -> bool:
        """Update ``[x,y,theta]`` and ``[vx,vy,omega]`` from a ROS message."""

        try:
            self._pose = _vector(pose, 3, "end_effector_pose")
            self._velocity = _vector(velocity, 3, "end_effector_velocity")
        except (TypeError, ValueError):
            self._pose_ok = False
            return False
        self._pose_ok = bool(valid)
        self._last_timestamp_s = float(self._clock())
        return self._pose_ok

    def update_wrench(self, wrench: ArrayLike, valid: bool = True) -> bool:
        """Update ``[Fx,Fy,Tz]`` from a ROS message."""

        try:
            self._wrench = _vector(wrench, 3, "wrench")
        except (TypeError, ValueError):
            self._wrench_ok = False
            return False
        self._wrench_ok = bool(valid)
        self._last_timestamp_s = float(self._clock())
        return self._wrench_ok

    def read_state(self) -> RobotState:
        return RobotState(
            timestamp_s=self._last_timestamp_s,
            joint_position=self._joint_position,
            joint_velocity=self._joint_velocity,
            end_effector_pose=self._pose,
            end_effector_velocity=self._velocity,
            wrench=self._wrench,
            sensor_ok=self._joint_ok and self._pose_ok and self._wrench_ok,
            source="ros2_hardware",
        )

    def write_admittance_parameters(self, command: AdmittanceCommand) -> None:
        self._last_command = command
        if self._command_sink is not None:
            self._command_sink(command)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def reset(self) -> None:
        self._last_command = None
        self._enabled = False


def apply_low_speed_limit(
    command: AdmittanceCommand,
    *,
    enabled: bool,
    velocity_scale_limit: float,
) -> AdmittanceCommand:
    """Apply the final low-speed cap to a parameter command.

    This function changes only the task-space velocity scale. It does not
    create or expose a joint-space or motor-level command.
    """

    if not enabled:
        return command
    if not np.isfinite(velocity_scale_limit) or velocity_scale_limit <= 0.0:
        raise ValueError("velocity_scale_limit must be finite and positive")
    limited = min(command.velocity_scale, float(velocity_scale_limit))
    return replace(command, velocity_scale=limited, low_speed_test=True)
