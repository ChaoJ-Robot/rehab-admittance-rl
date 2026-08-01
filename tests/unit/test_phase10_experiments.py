from __future__ import annotations

from pathlib import Path

import numpy as np
from rehab_sim.experiments.comparison import aggregate_rows, classical_policy
from rehab_sim.experiments.config import VALID_METHODS, load_phase10_config


def test_phase10_config_contains_five_methods_and_three_patients() -> None:
    root = Path(__file__).parents[2]
    config = load_phase10_config(root / "configs" / "phase10.yaml")
    assert config.methods == VALID_METHODS
    assert config.patient_profiles == ("mild", "moderate", "severe")
    assert config.seeds == (0, 1, 2, 3, 4)
    assert config.episodes_per_condition >= 1


def test_classical_comparison_policies_are_bounded_and_deterministic() -> None:
    root = Path(__file__).parents[2]
    config = load_phase10_config(root / "configs" / "phase10.yaml")
    observation = {
        "tracking_error": np.array([0.02, -0.01, 0.03]),
        "interaction_force": np.array([0.4, -0.1, 0.01]),
        "end_effector_velocity": np.array([0.04, 0.0, 0.0]),
    }
    for method in ("fixed_admittance", "rule_adaptive", "fuzzy_control"):
        policy = classical_policy(method, config.policy_parameters)
        first = policy(observation)
        second = policy(observation)
        assert first.shape == (4,)
        assert np.all(first >= -1.0)
        assert np.all(first <= 1.0)
        np.testing.assert_array_equal(first, second)


def test_phase10_summary_aggregates_success_and_safety_rates() -> None:
    from rehab_sim.experiments.comparison import EpisodeMetrics

    rows = [
        EpisodeMetrics(
            method="fixed_admittance",
            patient_profile="mild",
            task_name="point_to_point",
            seed=0,
            episode=0,
            reward=1.0,
            success=True,
            unsafe=False,
            length=10,
            mean_tracking_error=0.01,
            max_interaction_force=0.2,
            mean_interaction_force=0.1,
            patient_active_power_mean=0.01,
            patient_active_work_ratio_proxy=1.0,
            robot_assistance_energy=0.0,
            parameter_change_total=0.0,
            parameter_oscillation_rate=0.0,
            final_fatigue=0.1,
        ),
        EpisodeMetrics(
            method="fixed_admittance",
            patient_profile="mild",
            task_name="point_to_point",
            seed=1,
            episode=0,
            reward=0.0,
            success=False,
            unsafe=True,
            length=5,
            mean_tracking_error=0.02,
            max_interaction_force=0.3,
            mean_interaction_force=0.2,
            patient_active_power_mean=0.02,
            patient_active_work_ratio_proxy=0.8,
            robot_assistance_energy=0.01,
            parameter_change_total=0.1,
            parameter_oscillation_rate=0.2,
            final_fatigue=0.2,
        ),
    ]
    summary = aggregate_rows(rows)
    assert len(summary) == 1
    assert summary[0]["episodes"] == 2
    assert summary[0]["success_rate"] == 0.5
    assert summary[0]["unsafe_rate"] == 0.5
