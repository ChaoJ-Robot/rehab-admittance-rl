"""Deterministic, parameterized impedance-based virtual patient."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import pi

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rehab_sim.patients.profiles import PatientProfile
from rehab_sim.robot.kinematics import _vector, wrap_angle

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PatientOutput:
    """One virtual-patient interaction result."""

    force: FloatArray
    active_force: FloatArray
    directional_bias: FloatArray
    noise_force: FloatArray
    tremor_force: FloatArray
    delayed_reference_pose: FloatArray
    delayed_reference_velocity: FloatArray
    fatigue: float
    active_power_w: float
    effective_strength: float
    time_s: float


class ImpedancePatient:
    """Generate patient force from delayed reference tracking impedance.

    The model is intentionally not a human musculoskeletal simulation. Its
    output is an interaction wrench for a later simulation environment.
    It has no MuJoCo or actuator dependency and never controls the robot.
    """

    def __init__(self, profile: PatientProfile, sample_time_s: float, seed: int = 0) -> None:
        if sample_time_s <= 0.0:
            raise ValueError("sample_time_s must be positive")
        self.profile = profile
        self.sample_time_s = sample_time_s
        self.seed = seed
        self.delay_steps = int(round(profile.reaction_delay_ms / (1000.0 * sample_time_s)))
        self._history: deque[tuple[FloatArray, FloatArray]] = deque(maxlen=self.delay_steps + 1)
        self._rng = np.random.default_rng(seed)
        self._tremor_phase = np.zeros(3, dtype=np.float64)
        self._time_s = 0.0
        self._fatigue = 0.0
        self._reference_pose = np.zeros(3, dtype=np.float64)
        self._reference_velocity = np.zeros(3, dtype=np.float64)
        self.reset()

    @property
    def fatigue(self) -> float:
        """Current normalized fatigue in [0,1]."""

        return self._fatigue

    @property
    def time_s(self) -> float:
        """Current patient simulation time in seconds."""

        return self._time_s

    def reset(
        self,
        reference_pose: ArrayLike | None = None,
        reference_velocity: ArrayLike | None = None,
    ) -> None:
        """Reset time, fatigue, delay history and random disturbance phase."""

        self._time_s = 0.0
        self._fatigue = 0.0
        self._rng = np.random.default_rng(self.seed)
        self._tremor_phase[:] = self._rng.uniform(0.0, 2.0 * pi, size=3)
        self._reference_pose[:] = (
            np.zeros(3, dtype=np.float64)
            if reference_pose is None
            else _vector(reference_pose, 3, "reference_pose")
        )
        self._reference_velocity[:] = (
            np.zeros(3, dtype=np.float64)
            if reference_velocity is None
            else _vector(reference_velocity, 3, "reference_velocity")
        )
        self._history.clear()
        for _ in range(self.delay_steps + 1):
            self._history.append((self._reference_pose.copy(), self._reference_velocity.copy()))

    def _delayed_reference(
        self,
        reference_pose: FloatArray,
        reference_velocity: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        self._history.append((reference_pose.copy(), reference_velocity.copy()))
        delayed_pose, delayed_velocity = self._history[0]
        return delayed_pose.copy(), delayed_velocity.copy()

    def step(
        self,
        current_pose: ArrayLike,
        current_velocity: ArrayLike,
        reference_pose: ArrayLike,
        reference_velocity: ArrayLike | None = None,
        rest: bool = False,
    ) -> PatientOutput:
        """Generate one force sample and update fatigue.

        Args:
            current_pose: Current task pose ``[x,y,theta]`` in m, m, rad.
            current_velocity: Current task velocity in m/s, m/s, rad/s.
            reference_pose: Desired task pose in the same units.
            reference_velocity: Desired task velocity; defaults to zero.
            rest: Whether the patient is resting and can recover fatigue.
        """

        current = _vector(current_pose, 3, "current_pose")
        velocity = _vector(current_velocity, 3, "current_velocity")
        reference = _vector(reference_pose, 3, "reference_pose")
        reference_rate = (
            np.zeros(3, dtype=np.float64)
            if reference_velocity is None
            else _vector(reference_velocity, 3, "reference_velocity")
        )
        delayed_pose, delayed_velocity = self._delayed_reference(reference, reference_rate)
        pose_error = delayed_pose - current
        pose_error[2] = wrap_angle(float(pose_error[2]))
        velocity_error = delayed_velocity - velocity

        strength = self.profile.strength_scale * (1.0 - self._fatigue)
        coordination = self.profile.coordination_scale
        active_force = (
            strength
            * coordination
            * (
                self.profile.base_stiffness * pose_error
                + self.profile.base_damping * velocity_error
            )
        )
        bias = self.profile.directional_bias.copy()
        noise = self._rng.normal(0.0, self.profile.force_noise_std)
        tremor = self.profile.tremor_amplitude * np.sin(
            2.0 * pi * self.profile.tremor_frequency_hz * self._time_s + self._tremor_phase
        )
        force = active_force + bias + noise + tremor
        active_power = max(0.0, float(np.dot(active_force, velocity)))
        normalized_power = min(1.0, active_power / self.profile.power_normalization_w)
        fatigue_delta = self.profile.fatigue_rate * normalized_power
        if rest:
            fatigue_delta -= self.profile.recovery_rate
        self._fatigue = float(np.clip(self._fatigue + self.sample_time_s * fatigue_delta, 0.0, 1.0))
        output = PatientOutput(
            force=force.copy(),
            active_force=active_force.copy(),
            directional_bias=bias,
            noise_force=noise.copy(),
            tremor_force=tremor.copy(),
            delayed_reference_pose=delayed_pose,
            delayed_reference_velocity=delayed_velocity,
            fatigue=self._fatigue,
            active_power_w=active_power,
            effective_strength=strength,
            time_s=self._time_s,
        )
        self._time_s += self.sample_time_s
        return output
