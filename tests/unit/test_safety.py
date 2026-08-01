from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rehab_sim.config import load_yaml
from rehab_sim.safety import (
    SafePolicyRuntime,
    SafetyObservation,
    SafetySupervisor,
    load_safety_configuration,
)


@pytest.fixture
def supervisor() -> SafetySupervisor:
    root = Path(__file__).resolve().parents[2]
    admittance = load_yaml(root / "configs" / "admittance.yaml")
    safety = load_yaml(root / "configs" / "safety.yaml")
    return SafetySupervisor(load_safety_configuration(admittance, safety))


def _observation(**kwargs: object) -> SafetyObservation:
    values: dict[str, object] = {
        "interaction_wrench": np.zeros(3),
        "task_velocity": np.zeros(3),
    }
    values.update(kwargs)
    return SafetyObservation(**values)  # type: ignore[arg-type]


def test_action_projection_clips_and_rate_limits(supervisor: SafetySupervisor) -> None:
    current = np.array([3.0, 3.0, 0.25, 0.0, 1.0])
    decision = supervisor.supervise(np.array([10.0, -10.0, 10.0, -10.0]), current, _observation())
    config = supervisor.configuration
    assert decision.approved
    assert not decision.fallback
    assert np.all(decision.parameters >= config.parameter_lower)
    assert np.all(decision.parameters <= config.parameter_upper)
    assert np.all(np.abs(decision.parameters - current) <= config.parameter_rate_limits + 1.0e-12)
    assert decision.projection.action_clipped
    assert decision.projection.rate_limited


def test_nan_action_selects_fallback(supervisor: SafetySupervisor) -> None:
    current = np.array([3.0, 3.0, 0.25, 0.0, 1.0])
    decision = supervisor.supervise(np.array([np.nan, 0.0, 0.0, 0.0]), current, _observation())
    assert decision.fallback
    assert "non_finite_or_invalid_action" in decision.reasons
    np.testing.assert_array_equal(decision.parameters, supervisor.configuration.fallback_parameters)


def test_force_limit_selects_fallback(supervisor: SafetySupervisor) -> None:
    current = np.array([3.0, 3.0, 0.25, 0.0, 1.0])
    decision = supervisor.supervise(
        np.zeros(4),
        current,
        _observation(interaction_wrench=np.array([6.0, 0.0, 0.0])),
    )
    assert decision.fallback
    assert "interaction_force_near_limit" in decision.reasons


def test_nan_state_selects_fallback(supervisor: SafetySupervisor) -> None:
    decision = supervisor.supervise(
        np.zeros(4),
        np.array([3.0, 3.0, 0.25, 0.0, 1.0]),
        _observation(task_velocity=np.array([np.nan, 0.0, 0.0])),
    )
    assert decision.fallback
    assert "non_finite_state" in decision.reasons


def test_model_load_failure_falls_back(supervisor: SafetySupervisor, tmp_path: Path) -> None:
    model_path = tmp_path / "broken_model.zip"
    model_path.write_bytes(b"not-a-model")
    runtime = SafePolicyRuntime(supervisor)
    assert not runtime.load_model(model_path, lambda _: (_ for _ in ()).throw(RuntimeError("bad")))
    decision = runtime.act({}, np.array([3.0, 3.0, 0.25, 0.0, 1.0]), _observation())
    assert decision.fallback
    assert "model_load_failed" in decision.reasons
    assert "model_not_loaded" in decision.reasons


def test_policy_timeout_falls_back(supervisor: SafetySupervisor) -> None:
    clock_values = iter([0.0, 1.0])
    runtime = SafePolicyRuntime(
        supervisor,
        policy=lambda _: np.zeros(4),
        clock=lambda: next(clock_values),
    )
    decision = runtime.act({}, np.array([3.0, 3.0, 0.25, 0.0, 1.0]), _observation())
    assert decision.fallback
    assert "policy_timeout" in decision.reasons
