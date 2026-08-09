"""Clinical decision-support layer for the therapist console.

Turns the per-patient training history into two things a rehabilitation
physician actually needs:

1. A *longitudinal assessment*: how are score / completion rate / tracking
   error trending across recent sessions (improving / plateau / regressing).
2. A *progressive-loading prescription*: the next session's task, parameters
   and control mode, with an explicit rationale.

The layer is deterministic and LLM-free so it stays testable and keeps
working without any API key; the LLM layer only ever *expresses* these
results, it never invents clinical decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Classification = Literal["improving", "plateau", "regressing", "insufficient_data"]
DifficultyAction = Literal["upgrade", "maintain", "downgrade", "baseline"]

# Clinical difficulty ladder used by the progressive-loading rules.
TASK_DIFFICULTY: dict[str, int] = {
    "point_to_point": 1,
    "circle_tracking": 2,
    "figure8_tracking": 2,
    "follow_to_reach": 2,
    "visual_guided_reach": 2,
    "motion_intercept": 2,
    "maze_navigation": 3,
    "color_memory": 3,
    "marker_memory": 3,
}

# Candidate tasks when moving one rung up/down; the first one not seen in the
# two most recent sessions is preferred to add variety.
UPGRADE_TARGETS: dict[int, list[str]] = {
    1: ["follow_to_reach", "circle_tracking", "visual_guided_reach"],
    2: ["maze_navigation", "color_memory", "marker_memory", "motion_intercept"],
    3: [],
}
DOWNGRADE_TARGETS: dict[int, list[str]] = {
    3: ["follow_to_reach", "circle_tracking", "visual_guided_reach"],
    2: ["point_to_point"],
    1: [],
}

# Fallback defaults mirroring configs/tasks.yaml (used when the caller does
# not supply the live configuration table).
_FALLBACK_SPEED: dict[str, float] = {
    "point_to_point": 0.05,
    "circle_tracking": 0.10,
    "figure8_tracking": 0.10,
    "maze_navigation": 0.14,
    "follow_to_reach": 0.12,
    "visual_guided_reach": 0.12,
    "motion_intercept": 0.08,
}
_FALLBACK_DURATION: dict[str, float] = {
    "point_to_point": 4.0,
    "circle_tracking": 8.0,
    "figure8_tracking": 8.0,
    "maze_navigation": 12.0,
    "color_memory": 10.0,
    "follow_to_reach": 12.0,
    "visual_guided_reach": 12.0,
    "motion_intercept": 10.0,
    "marker_memory": 12.0,
}

# Classification thresholds on per-session slopes.
_COMPLETION_SLOPE_STEP = 0.02
_SCORE_SLOPE_STEP = 0.05
_MIN_SESSIONS = 3
_WINDOW = 8
_RECENT = 3

# Prescription thresholds on the recent average completion rate.
_UPGRADE_COMPLETION = 0.80
_DOWNGRADE_COMPLETION = 0.45
_RL_COMPLETION = 0.60

_SPEED_UPGRADE_FACTOR = 1.15
_SPEED_DOWNGRADE_FACTOR = 0.80


@dataclass(frozen=True)
class TrendAssessment:
    """Longitudinal trend over a patient's recent sessions."""

    sessions_analyzed: int
    classification: Classification
    score_slope: float
    completion_slope: float
    error_slope: float
    avg_score_recent: float | None
    avg_completion_recent: float | None
    avg_error_recent: float | None
    flags: list[str] = field(default_factory=list)
    narrative: str = ""
    risk_level: Literal["low", "moderate", "high"] = "low"
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SessionPrescription:
    """Agent suggestion for the next training session."""

    task: str
    task_params: dict[str, Any]
    mode: str
    difficulty_action: DifficultyAction
    rationale: list[str] = field(default_factory=list)
    risk_level: Literal["low", "moderate", "high"] = "low"
    confidence: float = 0.5
    missing_data: list[str] = field(default_factory=list)
    precautions: list[str] = field(default_factory=list)
    requires_doctor_approval: bool = True


def _slope(values: Sequence[float]) -> float:
    """Least-squares slope per session over the given series."""

    if len(values) < 2:
        return 0.0
    xs = np.arange(len(values), dtype=np.float64)
    ys = np.asarray(values, dtype=np.float64)
    return float(np.polyfit(xs, ys, 1)[0])


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def assess_history(
    history: Sequence[Any], window: int = _WINDOW, recent: int = _RECENT
) -> TrendAssessment:
    """Classify the longitudinal trend of one patient's session history.

    ``history`` items only need ``score`` / ``completion_rate`` /
    ``average_tracking_error`` attributes (PatientHistoryEntry-compatible).
    """

    entries = list(history)[-window:]
    if len(entries) < _MIN_SESSIONS:
        return TrendAssessment(
            sessions_analyzed=len(entries),
            classification="insufficient_data",
            score_slope=0.0,
            completion_slope=0.0,
            error_slope=0.0,
            avg_score_recent=_mean([entry.score for entry in entries]),
            avg_completion_recent=_mean([entry.completion_rate for entry in entries]),
            avg_error_recent=_mean([entry.average_tracking_error for entry in entries]),
            narrative="历史训练不足 3 次，暂无法评估趋势，建议先完成基线训练。",
            evidence=[f"已记录 {len(entries)} 次训练，少于趋势分析所需的 3 次"],
        )

    scores = [entry.score for entry in entries]
    completions = [entry.completion_rate for entry in entries]
    errors = [entry.average_tracking_error for entry in entries]
    score_slope = _slope(scores)
    completion_slope = _slope(completions)
    error_slope = _slope(errors)
    avg_score = _mean(scores[-recent:])
    avg_completion = _mean(completions[-recent:])
    avg_error = _mean(errors[-recent:])

    if completion_slope >= _COMPLETION_SLOPE_STEP or (
        score_slope > _SCORE_SLOPE_STEP and completion_slope >= 0.0
    ):
        classification: Classification = "improving"
    elif completion_slope <= -_COMPLETION_SLOPE_STEP:
        classification = "regressing"
    else:
        classification = "plateau"

    flags: list[str] = []
    if avg_completion is not None and avg_completion < _DOWNGRADE_COMPLETION:
        flags.append("近 3 次完成率偏低，任务难度可能超出当前能力")
    if avg_error is not None and avg_error > 0.03:
        flags.append("近 3 次平均轨迹误差偏大，建议关注运动控制质量")
    if classification == "regressing":
        flags.append("表现呈下滑趋势，建议复核患者状态并降低训练强度")
    recent_entries = entries[-recent:]
    recent_pain = [
        float(getattr(entry, "check_in", {}).get("pain_vas", 0.0)) for entry in recent_entries
    ]
    recent_fatigue = [
        float(getattr(entry, "check_in", {}).get("fatigue_0_10", 0.0)) for entry in recent_entries
    ]
    assistance = [float(getattr(entry, "robot_assistance_ratio", 0.0)) for entry in recent_entries]
    safety_count = sum(int(getattr(entry, "safety_trigger_count", 0)) for entry in recent_entries)
    collisions = sum(int(getattr(entry, "collision_count", 0)) for entry in recent_entries)
    risk_level: Literal["low", "moderate", "high"] = "low"
    if max(recent_pain, default=0.0) >= 7.0 or safety_count > 0:
        risk_level = "high"
    elif (
        max(recent_pain, default=0.0) >= 4.0
        or max(recent_fatigue, default=0.0) >= 6.0
        or _mean(assistance) is not None
        and (_mean(assistance) or 0.0) >= 0.65
        or collisions > 0
    ):
        risk_level = "moderate"
    if max(recent_pain, default=0.0) >= 4.0:
        flags.append("近期训练前疼痛评分偏高，建议先复核疼痛与肩部状态")
    if max(recent_fatigue, default=0.0) >= 6.0:
        flags.append("近期训练前疲劳较高，建议调整训练剂量或休息间隔")
    if assistance and (_mean(assistance) or 0.0) >= 0.65:
        flags.append("机器人辅助占比较高，完成率可能包含辅助代偿")
    if collisions > 0:
        flags.append(f"近 {len(recent_entries)} 次迷宫训练记录到 {collisions} 次碰撞")

    labels = {
        "improving": "持续改善",
        "plateau": "进入平台期",
        "regressing": "表现下滑",
    }
    narrative = (
        f"近 {len(entries)} 次训练{labels[classification]}："
        f"近 3 次平均完成率 {avg_completion:.0%}，平均轨迹误差 {avg_error:.4f}。"
    )
    evidence = [
        f"近 {len(recent_entries)} 次平均完成率 {avg_completion:.0%}",
        f"近 {len(recent_entries)} 次平均轨迹误差 {avg_error:.4f}",
        f"完成率每次变化斜率 {completion_slope:+.3f}",
    ]
    return TrendAssessment(
        sessions_analyzed=len(entries),
        classification=classification,
        score_slope=score_slope,
        completion_slope=completion_slope,
        error_slope=error_slope,
        avg_score_recent=avg_score,
        avg_completion_recent=avg_completion,
        avg_error_recent=avg_error,
        flags=flags,
        narrative=narrative,
        risk_level=risk_level,
        evidence=evidence,
    )


def _task_defaults(
    task: str, table: Mapping[str, Mapping[str, float]] | None
) -> tuple[float, float]:
    """Return (reference_speed, task_duration) defaults for one task."""

    if table and task in table:
        entry = table[task]
        return (
            float(entry.get("reference_speed", _FALLBACK_SPEED.get(task, 0.1))),
            float(entry.get("task_duration", _FALLBACK_DURATION.get(task, 8.0))),
        )
    return (
        _FALLBACK_SPEED.get(task, 0.1),
        _FALLBACK_DURATION.get(task, 8.0),
    )


def _pick_target(candidates: list[str], recent_tasks: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate not in recent_tasks:
            return candidate
    return candidates[0] if candidates else None


def _baseline_task(profile: str) -> str:
    return "circle_tracking" if profile == "mild" else "point_to_point"


def recommend_next_session(
    history: Sequence[Any],
    profile: str,
    task_defaults: Mapping[str, Mapping[str, float]] | None = None,
    clinical_context: Mapping[str, Any] | None = None,
) -> SessionPrescription:
    """Suggest the next session following progressive-loading principles.

    Completion rate ≥ 80% over the last three sessions advances the patient
    one rung on the difficulty ladder; below 45% steps back down; in between
    the current task is maintained. Speed scales with the decision, and RL
    mode is only suggested for stable, improving patients.
    """

    entries = list(history)
    clinical = clinical_context or {}
    missing_data: list[str] = []
    if not str(clinical.get("diagnosis", "")).strip():
        missing_data.append("诊断")
    if str(clinical.get("affected_side", "unknown")) == "unknown":
        missing_data.append("患侧")
    if not clinical.get("goals"):
        missing_data.append("康复目标")
    if not clinical.get("standardized_scores"):
        missing_data.append("标准化评估量表")
    precautions = [str(item) for item in clinical.get("precautions", [])]
    if not entries:
        task = _baseline_task(profile)
        speed, duration = _task_defaults(task, task_defaults)
        params: dict[str, Any] = {"task_duration": duration}
        if task in _FALLBACK_SPEED:
            params["reference_speed"] = round(speed, 2)
        return SessionPrescription(
            task=task,
            task_params=params,
            mode="fixed",
            difficulty_action="baseline",
            rationale=[
                "首次训练，建议以基线难度建立运动表现基准。",
                "采用固定导纳模式，便于观察患者原始能力水平。",
            ],
            risk_level="moderate" if precautions else "low",
            confidence=max(0.35, 0.65 - 0.07 * len(missing_data)),
            missing_data=missing_data,
            precautions=precautions,
        )

    assessment = assess_history(entries)
    last_task = entries[-1].task if entries[-1].task in TASK_DIFFICULTY else "point_to_point"
    difficulty = TASK_DIFFICULTY[last_task]
    recent_tasks = [entry.task for entry in entries[-2:]]
    avg_completion = assessment.avg_completion_recent
    classification = assessment.classification

    action: DifficultyAction
    target = last_task
    speed_factor = 1.0
    rationale: list[str] = []

    if (
        avg_completion is not None
        and avg_completion >= _UPGRADE_COMPLETION
        and classification != "regressing"
    ):
        upgraded = _pick_target(UPGRADE_TARGETS[difficulty], recent_tasks)
        if upgraded is not None:
            action = "upgrade"
            target = upgraded
            speed_factor = 1.0
            rationale.append(
                f"近 3 次平均完成率 {avg_completion:.0%}（≥{_UPGRADE_COMPLETION:.0%}），"
                "达到渐进加量标准，提高任务难度。"
            )
        else:
            action = "maintain"
            speed_factor = _SPEED_UPGRADE_FACTOR
            rationale.append(
                f"已处于最高难度等级，改为在任务内加量：参考速度提高至 "
                f"{_SPEED_UPGRADE_FACTOR:.0%}。"
            )
    elif avg_completion is not None and avg_completion < _DOWNGRADE_COMPLETION:
        downgraded = _pick_target(DOWNGRADE_TARGETS[difficulty], recent_tasks)
        if downgraded is not None and downgraded != last_task:
            action = "downgrade"
            target = downgraded
            speed_factor = _SPEED_DOWNGRADE_FACTOR
            rationale.append(
                f"近 3 次平均完成率 {avg_completion:.0%}（<{_DOWNGRADE_COMPLETION:.0%}），"
                "任务超出当前能力，退回较低难度并降低速度。"
            )
        else:
            action = "downgrade"
            speed_factor = _SPEED_DOWNGRADE_FACTOR
            rationale.append(
                f"近 3 次平均完成率 {avg_completion:.0%} 偏低，已为最低难度，"
                "改为降低参考速度以减少运动要求。"
            )
    else:
        action = "maintain"
        if avg_completion is not None:
            rationale.append(
                f"近 3 次平均完成率 {avg_completion:.0%}，处于适应区间，维持当前任务巩固运动学习。"
            )
        else:
            rationale.append("历史数据不足，维持当前任务。")

    if classification == "regressing":
        rationale.append("趋势评估显示表现下滑，本次不提高训练强度。")

    if assessment.risk_level == "high":
        action = "downgrade"
        target = last_task
        speed_factor = _SPEED_DOWNGRADE_FACTOR
        rationale.append("近期存在高风险信号，保持原任务并降低运动要求，训练前需医生复核。")

    speed, duration = _task_defaults(target, task_defaults)
    params = {"task_duration": duration}
    if target in _FALLBACK_SPEED:
        params["reference_speed"] = round(speed * speed_factor, 2)
    if target == "color_memory":
        last_length = 4
        if entries and getattr(entries[-1], "task", None) == "color_memory":
            last_length = 4  # session history does not record task params
        params["sequence_length"] = min(6, last_length + 1) if action == "upgrade" else last_length

    if (
        classification == "improving"
        and (avg_completion or 0.0) >= _RL_COMPLETION
        and assessment.risk_level == "low"
    ):
        mode = "rl"
        rationale.append("患者表现稳定且持续改善，建议启用 RL 参数调节实现辅助量个体化。")
    else:
        mode = "fixed"
        if action != "baseline":
            rationale.append("采用固定导纳模式，保持辅助输出稳定可预期。")

    return SessionPrescription(
        task=target,
        task_params=params,
        mode=mode,
        difficulty_action=action,
        rationale=rationale,
        risk_level=assessment.risk_level,
        confidence=min(0.9, max(0.4, 0.45 + 0.05 * len(entries) - 0.05 * len(missing_data))),
        missing_data=missing_data,
        precautions=precautions,
    )
