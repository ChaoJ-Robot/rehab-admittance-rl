"""Integration coverage for the three motor-cognitive tasks (thesis task set)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from backend.app.schemas.models import StartRequest  # noqa: E402
from backend.app.services.patient_store import PatientStore  # noqa: E402
from backend.app.services.session import TrainingSession  # noqa: E402


def _session(tmp_path: Path) -> TrainingSession:
    session = TrainingSession()
    session._patient_store = PatientStore(tmp_path)
    return session


def _start(session: TrainingSession, task: str, duration_s: float) -> None:
    request = StartRequest(
        task=task, patient_id="P090", duration_s=duration_s, mode="fixed"
    )
    asyncio.run(session.start(request))
    session._cancel_run_task()


def _tick(session: TrainingSession, seconds: float) -> None:
    for _ in range(int(seconds * session.refresh_hz)):
        if session._state != "running":
            break
        session._tick()


def test_visual_guided_reach_shows_only_current_target(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(session, "visual_guided_reach", 12.0)
    _tick(session, 0.5)
    telemetry = session.snapshot().telemetry
    assert telemetry is not None
    assert len(telemetry.task_targets) == 1, "guided mode must expose one target at a time"
    _tick(session, 30.0)
    assert session._state == "completed"
    assert session.snapshot().report is not None


def test_motion_intercept_shuttles_between_two_endpoints(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(session, "motion_intercept", 6.0)
    _tick(session, 0.2)
    first = session.snapshot().telemetry
    assert first is not None
    assert len(first.task_targets) == 2, "interception corridor needs two endpoints"
    first_reference = list(first.reference_pose)
    _tick(session, 1.0)
    second = session.snapshot().telemetry
    assert second is not None
    assert list(second.reference_pose) != first_reference, "target must keep moving"
    assert second.task_targets == first.task_targets
    _tick(session, 30.0)
    assert session._state == "completed"


def test_marker_memory_hides_marker_during_recall(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(session, "marker_memory", 12.0)
    _tick(session, 0.2)
    memorize = session.snapshot().telemetry
    assert memorize is not None
    assert memorize.task_phase == "memorize"
    assert len(memorize.memory_marker) == 3, "marker must be visible while memorising"
    _tick(session, 3.5)  # default memorize_duration is 3 s
    recall = session.snapshot().telemetry
    assert recall is not None
    assert recall.task_phase == "recall"
    assert recall.memory_marker == [], "marker must disappear during recall"
    _tick(session, 30.0)
    assert session._state == "completed"


def test_new_tasks_expose_adjustable_parameters(tmp_path: Path) -> None:
    session = _session(tmp_path)
    summary = session.config_summary()
    assert {"visual_guided_reach", "motion_intercept", "marker_memory"} <= set(summary.tasks)
    guided = {spec.name for spec in summary.task_params["visual_guided_reach"]}
    assert {"target_count", "target_radius", "reference_speed", "task_duration"} <= guided
    intercept = {spec.name for spec in summary.task_params["motion_intercept"]}
    assert {"reference_speed", "path_length", "task_duration"} <= intercept
    memory = {spec.name for spec in summary.task_params["marker_memory"]}
    assert {"marker_count", "memorize_duration", "task_duration"} <= memory
