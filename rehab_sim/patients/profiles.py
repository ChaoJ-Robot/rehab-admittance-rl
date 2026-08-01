"""Validated virtual-patient profile definitions and YAML loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rehab_sim.robot.kinematics import _vector

FloatArray = NDArray[np.float64]


def _triple(mapping: Mapping[str, Any], key: str) -> FloatArray:
    """Read a finite three-component patient vector."""

    if key not in mapping:
        raise ValueError(f"patient profile missing {key}")
    return _vector(mapping[key], 3, key)


@dataclass(frozen=True)
class PatientProfile:
    """Patient capability and disturbance parameters.

    Force vectors use ``[Fx,Fy,Tz]`` in N, N, N*m. Bias, noise and tremor
    values use the same units. Scale values are dimensionless.
    """

    name: str
    strength_scale: float
    coordination_scale: float
    reaction_delay_ms: float
    directional_bias: FloatArray
    force_noise_std: FloatArray
    tremor_amplitude: FloatArray
    tremor_frequency_hz: float
    fatigue_rate: float
    recovery_rate: float
    base_stiffness: FloatArray
    base_damping: FloatArray
    power_normalization_w: float

    def __post_init__(self) -> None:
        for field in (
            "directional_bias",
            "force_noise_std",
            "tremor_amplitude",
            "base_stiffness",
            "base_damping",
        ):
            value = _vector(getattr(self, field), 3, field)
            object.__setattr__(self, field, value)
        if not 0.0 <= self.strength_scale <= 1.0:
            raise ValueError("strength_scale must be in [0,1]")
        if not 0.0 <= self.coordination_scale <= 1.0:
            raise ValueError("coordination_scale must be in [0,1]")
        if not 0.0 <= self.reaction_delay_ms <= 500.0:
            raise ValueError("reaction_delay_ms must be in [0,500]")
        if self.tremor_frequency_hz < 0.0 or self.tremor_frequency_hz > 8.0:
            raise ValueError("tremor_frequency_hz must be in [0,8]")
        if self.fatigue_rate < 0.0 or self.recovery_rate < 0.0:
            raise ValueError("fatigue and recovery rates must be non-negative")
        for field in ("force_noise_std", "tremor_amplitude", "base_stiffness", "base_damping"):
            if np.any(getattr(self, field) < 0.0):
                raise ValueError(f"{field} must be non-negative")
        if np.any(self.base_stiffness <= 0.0) or np.any(self.base_damping <= 0.0):
            raise ValueError("base stiffness and damping must be positive")
        if self.power_normalization_w <= 0.0:
            raise ValueError("power_normalization_w must be positive")

    @classmethod
    def from_mapping(cls, name: str, mapping: Mapping[str, Any]) -> PatientProfile:
        """Create a validated profile from one YAML profile mapping."""

        return cls(
            name=name,
            strength_scale=float(mapping["strength_scale"]),
            coordination_scale=float(mapping["coordination_scale"]),
            reaction_delay_ms=float(mapping["reaction_delay_ms"]),
            directional_bias=_triple(mapping, "directional_bias"),
            force_noise_std=_triple(mapping, "force_noise_std"),
            tremor_amplitude=_triple(mapping, "tremor_amplitude"),
            tremor_frequency_hz=float(mapping["tremor_frequency_hz"]),
            fatigue_rate=float(mapping["fatigue_rate"]),
            recovery_rate=float(mapping["recovery_rate"]),
            base_stiffness=_triple(mapping, "base_stiffness"),
            base_damping=_triple(mapping, "base_damping"),
            power_normalization_w=float(mapping["power_normalization_w"]),
        )


def load_patient_profiles(config: Mapping[str, Any]) -> dict[str, PatientProfile]:
    """Load and validate all profiles from ``patient_profiles.yaml``."""

    raw_profiles = config.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        raise ValueError("patient profile config must contain a profiles mapping")
    profiles: dict[str, PatientProfile] = {}
    for name, raw_profile in raw_profiles.items():
        if not isinstance(name, str) or not isinstance(raw_profile, Mapping):
            raise ValueError("each patient profile must be a named mapping")
        profiles[name] = PatientProfile.from_mapping(name, raw_profile)
    return profiles
