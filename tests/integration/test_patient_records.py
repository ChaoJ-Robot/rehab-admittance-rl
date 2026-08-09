"""Integration coverage for per-patient records and new task variants."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from backend.app.schemas.models import StartRequest  # noqa: E402
from backend.app.services.patient_store import PatientStore, validate_patient_id  # noqa: E402
from backend.app.services.session import TrainingSession  # noqa: E402


def _session(tmp_path: Path) -> TrainingSession:
    """A session whose patient records live in a throwaway directory."""

    session = TrainingSession()
    session._patient_store = PatientStore(tmp_path)
    return session


def _start(session: TrainingSession, **kwargs: object) -> None:
    """Start a session on a fresh event loop and detach the loop task."""

    request = StartRequest(**kwargs)
    asyncio.run(session.start(request))
    session._cancel_run_task()


def _tick_until_complete(session: TrainingSession, seconds: float) -> None:
    """Fast-forward the simulation clock to completion."""

    steps = int(seconds * session.refresh_hz)
    for _ in range(steps):
        if session._state == "completed":
            break
        session._tick()


def test_patient_store_roundtrip(tmp_path: Path) -> None:
    store = PatientStore(tmp_path)
    record = store.create("P001", "moderate")
    record.latest_parameters = [3.0, 3.0, 0.25, 0.5, 1.0]
    store.save(record)

    loaded = store.load("P001")
    assert loaded is not None
    assert loaded.patient_id == "P001"
    assert loaded.profile == "moderate"
    assert loaded.latest_parameters == [3.0, 3.0, 0.25, 0.5, 1.0]
    assert store.list_ids() == ["P001"]
    assert store.load("missing") is None


def test_patient_id_validation() -> None:
    assert validate_patient_id("P001")
    assert validate_patient_id("a-b_c9")
    assert not validate_patient_id("")
    assert not validate_patient_id("bad/id")
    assert not validate_patient_id("x" * 33)


def test_register_patient_and_summary(tmp_path: Path) -> None:
    session = _session(tmp_path)
    summary = session.register_patient("P007", "severe")
    assert summary.patient_id == "P007"
    assert summary.profile == "severe"
    assert summary.session_count == 0

    fetched = session.patient_summary("P007")
    assert fetched is not None
    assert fetched.profile == "severe"
    assert [item.patient_id for item in session.list_patients()] == ["P007"]


def test_invalid_patient_id_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path)
    with pytest.raises(ValueError):
        _start(session, task="point_to_point", patient_id="bad/id")


def test_follow_to_reach_builds_targets(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(
        session,
        task="follow_to_reach",
        patient_id="P001",
        task_params={"target_count": 3, "reference_speed": 0.15},
    )
    assert session._follow is not None
    assert len(session._follow.targets) == 3

    # Moving legs have non-zero velocity, dwell stages hold still.
    plan = session._follow
    leg_s = plan._leg_times[0]
    moving_pose, moving_velocity = plan.reference(leg_s / 2)
    assert np.linalg.norm(moving_velocity) > 0.0
    dwelling_pose, dwelling_velocity = plan.reference(leg_s + 0.05)
    assert np.linalg.norm(dwelling_velocity) == 0.0
    assert np.allclose(dwelling_pose, plan.targets[0])
    assert plan.progress(leg_s + 0.05) < 1.0


def test_maze_map_selection(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(session, task="maze_navigation", patient_id="P001", task_params={"map": "spiral"})
    assert session._maze is not None
    spiral_walls = [tuple(wall) for wall in session._maze_walls]

    _start(session, task="maze_navigation", patient_id="P001", task_params={"map": "ladder"})
    assert session._maze is not None
    ladder_walls = [tuple(wall) for wall in session._maze_walls]
    assert ladder_walls != spiral_walls

    with pytest.raises(ValueError):
        _start(session, task="maze_navigation", patient_id="P001", task_params={"map": "void"})


def test_task_params_override_speed(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(
        session,
        task="circle_tracking",
        patient_id="P001",
        task_params={"reference_speed": 0.04},
    )
    assert session._task is not None
    assert session._task.reference_speed_m_per_s == pytest.approx(0.04)


def test_color_memory_sequence_length_override(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(
        session,
        task="color_memory",
        patient_id="P001",
        task_params={"sequence_length": 3, "memorize_duration": 2.0},
    )
    assert session._color is not None
    assert len(session._color.sequence) == 3
    assert session._color.memorize_s == pytest.approx(2.0)


def test_parameters_resume_across_sessions(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(
        session,
        task="point_to_point",
        patient_id="P001",
        mode="rl",
        duration_s=1.0,
    )
    _tick_until_complete(session, 1.2)
    assert session._state == "completed"
    record = session._patient_store.load("P001")
    assert record is not None
    assert record.session_count == 1
    stored = record.latest_parameters
    assert len(stored) == 5

    # A second session for the same patient resumes the stored parameters.
    # Fixed mode keeps them untouched, so the comparison is exact.
    _start(session, task="circle_tracking", patient_id="P001", mode="fixed")
    assert np.allclose(session._parameters, stored)

    # A brand-new patient still starts from the generic baseline.
    _start(session, task="circle_tracking", patient_id="P002", mode="fixed")
    assert np.allclose(session._parameters, session._baseline)


def test_parameters_are_clipped_to_safety_bounds(tmp_path: Path) -> None:
    session = _session(tmp_path)
    store = session._patient_store
    record = store.create("P009", "moderate")
    record.latest_parameters = [999.0, 3.0, 0.25, -5.0, 1.0]
    store.save(record)

    _start(session, task="point_to_point", patient_id="P009", mode="rl")
    assert session._parameters[0] <= 10.0
    assert session._parameters[3] >= 0.0


def test_stopped_session_still_persists(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(session, task="follow_to_reach", patient_id="P003", mode="rl")
    session._parameters = np.asarray([4.0, 4.0, 0.3, 0.8, 1.0], dtype=np.float64)
    session.stop()
    record = session._patient_store.load("P003")
    assert record is not None
    assert record.session_count == 1
    assert record.history[0].task == "follow_to_reach"
    assert time.time() - record.last_session_at < 5.0


def test_clinical_profile_and_check_in_are_persisted(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.register_patient("P030", "moderate")
    updated = session.update_clinical_profile(
        "P030",
        {
            "diagnosis": "脑卒中后上肢功能障碍",
            "affected_side": "left",
            "rehab_stage": "subacute",
            "goals": ["提高主动到达能力"],
            "precautions": ["肩痛时停止"],
            "standardized_scores": {"FMA-UE": 31},
        },
    )
    assert updated is not None
    assert updated.clinical_profile.affected_side == "left"
    _start(
        session,
        task="point_to_point",
        patient_id="P030",
        duration_s=1.0,
        check_in={"pain_vas": 3, "fatigue_0_10": 4, "exertion_rpe": 2},
    )
    _tick_until_complete(session, 2.0)
    record = session._patient_store.load("P030")
    assert record is not None
    assert record.history[-1].check_in["pain_vas"] == 3
    assert "active_participation_ratio" in record.history[-1].__dict__


def test_procedural_maze_uses_observed_progress_and_reports_quality(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(
        session,
        task="maze_navigation",
        patient_id="P031",
        duration_s=60.0,
        task_params={"map": "procedural_easy", "maze_seed": 9, "reference_speed": 0.24},
    )
    assert len(session._maze_walls) >= 40
    _tick_until_complete(session, 60.0)
    assert session._state == "completed"
    report = session.snapshot().report
    assert report is not None
    assert report.path_efficiency is not None
    assert 0.0 < report.path_efficiency <= 1.0
    assert report.target_hit_count > 0


def test_maze_timeout_is_not_recorded_as_success(tmp_path: Path) -> None:
    session = _session(tmp_path)
    _start(
        session,
        task="maze_navigation",
        patient_id="P032",
        duration_s=4.0,
        task_params={"map": "procedural_challenge", "maze_seed": 1},
    )
    for _ in range(int(5 * session.refresh_hz)):
        if session._state != "running":
            break
        session._tick()
    assert session._state == "stopped"
    assert session.snapshot().report is not None
    assert not session.snapshot().report.completed


def test_unknown_task_parameter_is_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path)
    with pytest.raises(ValueError, match="unsupported task parameters"):
        _start(
            session,
            task="point_to_point",
            patient_id="P033",
            task_params={"unsafe_gain": 99},
        )
