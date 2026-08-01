"""Deterministic point, circle and figure-eight task references."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import cos, sin

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rehab_sim.robot.kinematics import _vector

FloatArray = NDArray[np.float64]


class ReferenceTrajectory(ABC):
    """Interface shared by Phase 4 reference tasks."""

    duration_s: float
    success_tolerance_m: float

    @abstractmethod
    def reference(self, time_s: float) -> tuple[FloatArray, FloatArray]:
        """Return reference pose and velocity at time ``time_s``."""

    def progress(self, time_s: float) -> float:
        """Return normalized task progress in [0,1]."""

        return float(np.clip(time_s / self.duration_s, 0.0, 1.0))

    def complete(self, time_s: float) -> bool:
        """Return whether the configured task duration has elapsed."""

        return time_s >= self.duration_s

    def success(self, time_s: float, pose_error: ArrayLike) -> bool:
        """Return whether completion occurred with acceptable XY error."""

        error = _vector(pose_error, 3, "pose_error")
        return (
            self.complete(time_s) and float(np.linalg.norm(error[:2])) <= self.success_tolerance_m
        )


@dataclass
class PointToPointTrajectory(ReferenceTrajectory):
    """Linear point-to-point task along the world +X direction."""

    start_pose: FloatArray
    target_distance_m: float
    duration_s: float
    success_tolerance_m: float

    def __post_init__(self) -> None:
        self.start_pose = _vector(self.start_pose, 3, "start_pose")
        if self.target_distance_m <= 0.0 or self.duration_s <= 0.0:
            raise ValueError("point task distance and duration must be positive")

    def reference(self, time_s: float) -> tuple[FloatArray, FloatArray]:
        progress = self.progress(time_s)
        pose = self.start_pose.copy()
        pose[0] += self.target_distance_m * progress
        velocity = np.zeros(3, dtype=np.float64)
        if time_s < self.duration_s:
            velocity[0] = self.target_distance_m / self.duration_s
        return pose, velocity


@dataclass
class CircleTrajectory(ReferenceTrajectory):
    """Planar circular path starting at the supplied pose."""

    start_pose: FloatArray
    radius_m: float
    reference_speed_m_per_s: float
    duration_s: float
    success_tolerance_m: float

    def __post_init__(self) -> None:
        self.start_pose = _vector(self.start_pose, 3, "start_pose")
        if self.radius_m <= 0.0 or self.reference_speed_m_per_s <= 0.0 or self.duration_s <= 0.0:
            raise ValueError("circle radius, speed and duration must be positive")

    def reference(self, time_s: float) -> tuple[FloatArray, FloatArray]:
        omega = self.reference_speed_m_per_s / self.radius_m
        phase = omega * min(time_s, self.duration_s)
        center = self.start_pose.copy()
        center[1] -= self.radius_m
        pose = center.copy()
        pose[0] += self.radius_m * sin(phase)
        pose[1] += self.radius_m * cos(phase)
        pose[2] += 0.1 * sin(phase)
        velocity = np.zeros(3, dtype=np.float64)
        if time_s < self.duration_s:
            velocity[:] = [
                self.radius_m * omega * cos(phase),
                -self.radius_m * omega * sin(phase),
                0.1 * omega * cos(phase),
            ]
        return pose, velocity


@dataclass
class FigureEightTrajectory(ReferenceTrajectory):
    """Planar lemniscate path starting at ``start_pose``."""

    start_pose: FloatArray
    width_m: float
    reference_speed_m_per_s: float
    duration_s: float
    success_tolerance_m: float

    def __post_init__(self) -> None:
        self.start_pose = _vector(self.start_pose, 3, "start_pose")
        if self.width_m <= 0.0 or self.reference_speed_m_per_s <= 0.0 or self.duration_s <= 0.0:
            raise ValueError("figure-eight width, speed and duration must be positive")

    def reference(self, time_s: float) -> tuple[FloatArray, FloatArray]:
        amplitude_x = self.width_m / 2.0
        amplitude_y = self.width_m / 3.0
        omega = self.reference_speed_m_per_s / max(amplitude_x, 1.0e-6)
        phase = omega * min(time_s, self.duration_s)
        pose = self.start_pose.copy()
        pose[0] += amplitude_x * sin(phase)
        pose[1] += amplitude_y * sin(phase) * cos(phase)
        pose[2] += 0.1 * sin(phase)
        velocity = np.zeros(3, dtype=np.float64)
        if time_s < self.duration_s:
            velocity[:] = [
                amplitude_x * omega * cos(phase),
                amplitude_y * omega * cos(2.0 * phase),
                0.1 * omega * cos(phase),
            ]
        return pose, velocity
