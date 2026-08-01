"""Damped least-squares task-pose to joint-target mapping for simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rehab_sim.robot.kinematics import Planar3RGeometry, _vector, wrap_angle

FloatArray = NDArray[np.float64]


@dataclass
class DampedLeastSquaresIK:
    """Bounded iterative inverse kinematics used by the Phase 2 baseline.

    This is a simulation-side target mapper. It does not generate motor
    torques or replace the robot's low-level actuator controller.
    """

    geometry: Planar3RGeometry
    joint_lower: FloatArray
    joint_upper: FloatArray
    damping: float = 0.02
    step_scale: float = 0.8
    max_iterations: int = 30

    def __post_init__(self) -> None:
        self.joint_lower = _vector(self.joint_lower, 3, "joint_lower")
        self.joint_upper = _vector(self.joint_upper, 3, "joint_upper")
        if np.any(self.joint_lower > self.joint_upper):
            raise ValueError("joint_lower must not exceed joint_upper")
        if self.damping <= 0.0 or self.step_scale <= 0.0 or self.max_iterations < 1:
            raise ValueError("IK damping, step_scale and max_iterations are invalid")

    def solve(self, target_pose: ArrayLike, initial_qpos: ArrayLike) -> FloatArray:
        """Return a joint target that minimizes task-pose error."""

        target = _vector(target_pose, 3, "target_pose")
        qpos = np.clip(_vector(initial_qpos, 3, "initial_qpos"), self.joint_lower, self.joint_upper)
        for _ in range(self.max_iterations):
            current = self.geometry.forward(qpos)
            error = target - current
            error[2] = wrap_angle(float(error[2]))
            if np.linalg.norm(error) < 1.0e-7:
                break
            jacobian = self.geometry.jacobian(qpos)
            regularized = jacobian @ jacobian.T + (self.damping**2) * np.eye(3)
            delta = jacobian.T @ np.linalg.solve(regularized, error)
            qpos = np.clip(qpos + self.step_scale * delta, self.joint_lower, self.joint_upper)
        return qpos
