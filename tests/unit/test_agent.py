from __future__ import annotations

from pathlib import Path

import pytest
from rehab_sim.agent import AgentObservation, RuleBasedAgent, load_agent_config


def _agent() -> RuleBasedAgent:
    root = Path(__file__).resolve().parents[2]
    return RuleBasedAgent(load_agent_config(root / "configs" / "agent.yaml"))


def _observation(
    elapsed_s: float,
    *,
    error: float = 0.005,
    force: float = 0.1,
    speed: float = 0.05,
    power: float = 0.02,
    fatigue: float = 0.1,
) -> AgentObservation:
    return AgentObservation(
        task="point_to_point",
        elapsed_s=elapsed_s,
        tracking_error_norm=error,
        interaction_force_norm=force,
        task_speed_norm=speed,
        human_power_w=power,
        fatigue=fatigue,
        task_progress=min(1.0, elapsed_s / 4.0),
        safety_status="safe",
    )


def test_rule_agent_detects_error_force_and_fatigue_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = _agent()
    caplog.set_level("INFO", logger="rehab.agent")
    started = agent.start("point_to_point")
    assert started is not None
    error_event = agent.observe(_observation(2.1, error=0.05))
    force_event = agent.observe(_observation(4.2, force=0.6))
    fatigue_event = agent.observe(_observation(6.3, fatigue=0.8))
    assert error_event is not None and error_event.event == "tracking_error_high"
    assert force_event is not None and force_event.event == "force_too_high"
    assert fatigue_event is not None and fatigue_event.event == "fatigue_detected"
    assert len(agent.events) >= 4
    assert "agent_event" in caplog.text


def test_rule_agent_detects_inactivity_and_builds_summary() -> None:
    agent = _agent()
    agent.start("point_to_point")
    assert agent.observe(_observation(0.0, power=0.0)) is not None
    assert agent.observe(_observation(0.6, power=0.0)) is None
    inactive = agent.observe(_observation(1.2, power=0.0))
    assert inactive is not None and inactive.event == "patient_inactive"
    event, summary = agent.complete(
        _observation(2.0),
        {
            "completed": True,
            "average_tracking_error": 0.01,
            "peak_interaction_force": 0.3,
            "patient_active_work": 0.2,
        },
    )
    assert event is not None and event.event == "task_completed"
    assert summary.title == "训练总结"
    assert summary.event_count == len(agent.events)


def test_agent_speech_failure_does_not_raise() -> None:
    config = load_agent_config(Path(__file__).resolve().parents[2] / "configs" / "agent.yaml")
    config = config.__class__(**{**config.__dict__, "speech_enabled": True})

    def broken_speech(_: str) -> None:
        raise RuntimeError("audio unavailable")

    agent = RuleBasedAgent(config, speech_sink=broken_speech)
    event = agent.start("point_to_point")
    assert event is not None
