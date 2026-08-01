from pathlib import Path

import numpy as np
from rehab_sim.config import load_yaml
from rehab_sim.patients import ImpedancePatient, load_patient_profiles


def _profiles():
    root = Path(__file__).parents[2]
    return load_patient_profiles(load_yaml(root / "configs" / "patient_profiles.yaml"))


def test_three_profiles_are_loaded_and_have_distinct_active_strength() -> None:
    profiles = _profiles()
    assert set(profiles) == {"mild", "moderate", "severe"}
    forces = {}
    for name, profile in profiles.items():
        patient = ImpedancePatient(profile, sample_time_s=0.01, seed=1)
        output = None
        for _ in range(patient.delay_steps + 1):
            output = patient.step(
                current_pose=np.zeros(3),
                current_velocity=np.zeros(3),
                reference_pose=np.array([0.1, 0.0, 0.0]),
            )
        assert output is not None
        forces[name] = output.active_force[0]

    assert forces["mild"] > forces["moderate"] > forces["severe"]


def test_same_seed_is_reproducible_and_different_seed_changes_noise() -> None:
    profile = _profiles()["moderate"]
    patient_a = ImpedancePatient(profile, sample_time_s=0.01, seed=11)
    patient_b = ImpedancePatient(profile, sample_time_s=0.01, seed=11)
    patient_c = ImpedancePatient(profile, sample_time_s=0.01, seed=12)
    kwargs = {
        "current_pose": np.zeros(3),
        "current_velocity": np.array([0.05, 0.0, 0.0]),
        "reference_pose": np.array([0.1, 0.02, 0.0]),
        "reference_velocity": np.array([0.05, 0.0, 0.0]),
    }
    output_a = [patient_a.step(**kwargs).force for _ in range(20)]
    output_b = [patient_b.step(**kwargs).force for _ in range(20)]
    output_c = [patient_c.step(**kwargs).force for _ in range(20)]

    np.testing.assert_array_equal(output_a, output_b)
    assert not np.array_equal(output_a, output_c)


def test_reaction_delay_holds_old_reference_before_release() -> None:
    profile = _profiles()["moderate"]
    patient = ImpedancePatient(profile, sample_time_s=0.01, seed=2)
    delayed = []
    for _ in range(patient.delay_steps + 1):
        output = patient.step(np.zeros(3), np.zeros(3), np.array([0.2, 0.0, 0.0]))
        delayed.append(output.delayed_reference_pose[0])

    assert all(value == 0.0 for value in delayed[:-1])
    assert delayed[-1] == 0.2


def test_fatigue_increases_with_active_power_and_recovers_at_rest() -> None:
    patient = ImpedancePatient(_profiles()["mild"], sample_time_s=0.01, seed=3)
    initial_fatigue = patient.fatigue
    active_outputs = [
        patient.step(
            current_pose=np.zeros(3),
            current_velocity=np.array([0.1, 0.0, 0.0]),
            reference_pose=np.array([0.2, 0.0, 0.0]),
            reference_velocity=np.zeros(3),
        )
        for _ in range(1000)
    ]
    active_fatigue = active_outputs[-1].fatigue
    active_power = max(output.active_power_w for output in active_outputs)
    assert initial_fatigue == 0.0
    assert active_power > 0.0
    assert active_fatigue > initial_fatigue

    recovered = [
        patient.step(
            current_pose=np.zeros(3),
            current_velocity=np.zeros(3),
            reference_pose=np.zeros(3),
            rest=True,
        )
        for _ in range(100)
    ][-1]
    assert recovered.fatigue < active_fatigue
