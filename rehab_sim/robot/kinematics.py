"""Forward kinematics for the planar three-revolute-joint robot.

All positions use metres, all angles use radians, and the task pose is
``[x, y, theta]`` in the MuJoCo world frame. The three joint axes are +Z, so
the planar Jacobian maps joint velocity [rad/s] to [m/s, m/s, rad/s].
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import cos, pi, sin
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    """Convert a value to a finite one-dimensional vector of a fixed size."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to the half-open interval [-pi, pi)."""

    return (angle + pi) % (2.0 * pi) - pi


def _rotate(angle: float, vector: FloatArray) -> FloatArray:
    """Rotate an XY vector by ``angle`` radians about +Z."""

    c = cos(angle)
    s = sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


@dataclass(frozen=True)
class WorkspaceBounds:
    """Axis-aligned task-space bounds in metres and radians."""

    x: tuple[float, float]
    y: tuple[float, float]
    theta: tuple[float, float]

    def contains(self, pose: ArrayLike) -> bool:
        """Return whether ``pose=[x,y,theta]`` is inside all configured bounds."""

        value = _vector(pose, 3, "pose")
        return bool(
            self.x[0] <= value[0] <= self.x[1]
            and self.y[0] <= value[1] <= self.y[1]
            and self.theta[0] <= value[2] <= self.theta[1]
        )


@dataclass(frozen=True)
class Planar3RGeometry:
    """CAD-derived geometry for the serial planar 3R mechanism.

    ``joint1_origin`` is expressed in the world XY frame. The other two
    offsets and ``tool_offset`` are expressed in their respective parent body
    frames at the zero pose. At zero joint angles the three link centers are
    collinear according to the imported CAD assembly.
    """

    joint1_origin: FloatArray
    link2_offset: FloatArray
    link3_offset: FloatArray
    tool_offset: FloatArray

    def __post_init__(self) -> None:
        for name in ("joint1_origin", "link2_offset", "link3_offset", "tool_offset"):
            value = _vector(getattr(self, name), 2, name)
            object.__setattr__(self, name, value)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Planar3RGeometry:
        """Build geometry from the ``kinematics`` section of ``robot.yaml``."""

        raw = config.get("kinematics")
        if not isinstance(raw, Mapping):
            raise ValueError("robot configuration must contain a kinematics mapping")
        required = ("joint1_origin_m", "link2_offset_m", "link3_offset_m", "tool_offset_m")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"robot kinematics is missing: {', '.join(missing)}")
        return cls(
            joint1_origin=_vector(raw["joint1_origin_m"], 2, "joint1_origin_m"),
            link2_offset=_vector(raw["link2_offset_m"], 2, "link2_offset_m"),
            link3_offset=_vector(raw["link3_offset_m"], 2, "link3_offset_m"),
            tool_offset=_vector(raw["tool_offset_m"], 2, "tool_offset_m"),
        )

    def joint_positions(self, qpos: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return world XY positions of joints 1, 2 and 3 for ``qpos``."""

        q = _vector(qpos, 3, "qpos")
        angle1 = q[0]
        angle2 = q[0] + q[1]
        joint1 = self.joint1_origin.copy()
        joint2 = joint1 + _rotate(angle1, self.link2_offset)
        joint3 = joint2 + _rotate(angle2, self.link3_offset)
        return joint1, joint2, joint3

    def forward(self, qpos: ArrayLike) -> FloatArray:
        """Return end-effector task pose ``[x, y, theta]`` for joint angles."""

        q = _vector(qpos, 3, "qpos")
        joint1, _, joint3 = self.joint_positions(q)
        del joint1  # The first joint position is returned by joint_positions.
        tool_position = joint3 + _rotate(float(q.sum()), self.tool_offset)
        return np.array([tool_position[0], tool_position[1], wrap_angle(float(q.sum()))])

    def jacobian(self, qpos: ArrayLike) -> FloatArray:
        """Return the 3x3 task Jacobian at ``qpos``."""

        q = _vector(qpos, 3, "qpos")
        angles = np.cumsum(q)
        segments = (self.link2_offset, self.link3_offset, self.tool_offset)
        jacobian = np.zeros((3, 3), dtype=np.float64)
        for joint_index in range(3):
            for segment_index in range(joint_index, 3):
                rotated = _rotate(float(angles[segment_index]), segments[segment_index])
                jacobian[0, joint_index] -= rotated[1]
                jacobian[1, joint_index] += rotated[0]
            jacobian[2, joint_index] = 1.0
        return jacobian
