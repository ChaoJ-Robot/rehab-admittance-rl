"""Unit coverage for the clinical decision-support agent layer."""

from __future__ import annotations

import time

from backend.app.services.patient_store import PatientHistoryEntry
from rehab_sim.agent.clinical import (
    TASK_DIFFICULTY,
    assess_history,
    recommend_next_session,
)


def _entry(completion: float, score: float, error: float = 0.008, task: str = "point_to_point"):
    """One PatientHistoryEntry-compatible record."""

    return PatientHistoryEntry(
        session_id=f"s{time.time_ns()}",
        task=task,
        timestamp=time.time(),
        duration_s=4.0,
        score=score,
        completion_rate=completion,
        average_tracking_error=error,
    )


def test_assess_insufficient_data() -> None:
    assessment = assess_history([_entry(0.5, 0.0), _entry(0.6, 0.2)])
    assert assessment.classification == "insufficient_data"
    assert assessment.sessions_analyzed == 2


def test_assess_improving() -> None:
    entries = [_entry(c, s) for c, s in [(0.3, -2.0), (0.5, -1.0), (0.7, 0.5), (0.85, 1.2)]]
    assessment = assess_history(entries)
    assert assessment.classification == "improving"
    assert assessment.completion_slope > 0.0
    assert assessment.narrative


def test_assess_regressing_with_flags() -> None:
    entries = [
        _entry(0.8, 1.0, 0.01, "maze_navigation"),
        _entry(0.6, 0.2, 0.02, "maze_navigation"),
        _entry(0.4, -0.5, 0.035, "maze_navigation"),
        _entry(0.25, -1.5, 0.04, "maze_navigation"),
    ]
    assessment = assess_history(entries)
    assert assessment.classification == "regressing"
    assert any("偏低" in flag for flag in assessment.flags)
    assert any("下滑" in flag for flag in assessment.flags)


def test_assess_plateau() -> None:
    entries = [_entry(c, 0.5) for c in (0.60, 0.61, 0.60, 0.62)]
    assessment = assess_history(entries)
    assert assessment.classification == "plateau"


def test_prescription_first_session_baseline() -> None:
    severe = recommend_next_session([], "severe")
    assert severe.task == "point_to_point"
    assert severe.mode == "fixed"
    assert severe.difficulty_action == "baseline"
    mild = recommend_next_session([], "mild")
    assert mild.task == "circle_tracking"


def test_prescription_upgrade_advances_difficulty() -> None:
    entries = [_entry(c, s) for c, s in [(0.7, 0.3), (0.82, 0.9), (0.88, 1.4), (0.92, 1.8)]]
    prescription = recommend_next_session(entries, "moderate")
    assert prescription.difficulty_action == "upgrade"
    assert TASK_DIFFICULTY[prescription.task] == 2
    assert prescription.mode == "rl"  # stable + improving + completion >= 0.6


def test_prescription_maintain_within_adaptation_band() -> None:
    entries = [_entry(c, 0.5) for c in (0.60, 0.61, 0.60, 0.62)]
    prescription = recommend_next_session(entries, "moderate")
    assert prescription.difficulty_action == "maintain"
    assert prescription.task == "point_to_point"
    assert prescription.mode == "fixed"


def test_prescription_downgrade_on_low_completion() -> None:
    entries = [
        _entry(0.8, 1.0, 0.01, "maze_navigation"),
        _entry(0.6, 0.2, 0.02, "maze_navigation"),
        _entry(0.4, -0.5, 0.03, "maze_navigation"),
        _entry(0.25, -1.5, 0.04, "maze_navigation"),
    ]
    prescription = recommend_next_session(entries, "moderate")
    assert prescription.difficulty_action == "downgrade"
    assert TASK_DIFFICULTY[prescription.task] == 2
    assert prescription.mode == "fixed"
    # speed reduced below the fallback default of the chosen task
    assert prescription.task_params["reference_speed"] < 0.12


def test_prescription_top_ladder_intensifies_within_task() -> None:
    entries = [_entry(c, 1.0, 0.008, "maze_navigation") for c in (0.85, 0.90, 0.92, 0.95)]
    prescription = recommend_next_session(entries, "moderate")
    assert prescription.task == "maze_navigation"
    assert prescription.task_params["reference_speed"] > 0.14  # intensified


def test_prescription_uses_provided_defaults() -> None:
    entries = [_entry(c, s) for c, s in [(0.7, 0.3), (0.82, 0.9), (0.88, 1.4), (0.92, 1.8)]]
    table = {task: {"reference_speed": 0.2, "task_duration": 30.0} for task in TASK_DIFFICULTY}
    prescription = recommend_next_session(entries, "moderate", table)
    assert prescription.task_params["reference_speed"] == 0.2
    assert prescription.task_params["task_duration"] == 30.0


def test_readiness_and_assistance_raise_clinical_risk() -> None:
    entries = [_entry(0.8, 1.0) for _ in range(3)]
    for entry in entries:
        entry.check_in = {"pain_vas": 5.0, "fatigue_0_10": 7.0}
        entry.robot_assistance_ratio = 0.8
    assessment = assess_history(entries)
    assert assessment.risk_level == "moderate"
    assert any("疼痛" in flag for flag in assessment.flags)
    prescription = recommend_next_session(
        entries,
        "moderate",
        clinical_context={"diagnosis": "卒中", "affected_side": "left", "goals": ["到达"]},
    )
    assert prescription.mode == "fixed"
    assert prescription.requires_doctor_approval
    assert "标准化评估量表" in prescription.missing_data
