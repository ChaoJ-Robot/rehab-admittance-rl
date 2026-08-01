"""Rule-driven, control-independent feedback Agent for Phase 8."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rehab_sim.config import load_yaml

LOGGER = logging.getLogger("rehab.agent")
Severity = Literal["info", "positive", "warning", "critical"]


@dataclass(frozen=True)
class AgentConfig:
    """Validated thresholds for rule-based feedback."""

    enabled: bool
    mode: str
    speech_enabled: bool
    feedback_cooldown_s: float
    tracking_error_high_m: float
    tracking_error_good_m: float
    interaction_force_high_n: float
    task_speed_fast: float
    task_speed_slow: float
    human_power_low_w: float
    inactivity_duration_s: float
    fatigue_high: float
    hardware_validation_required: bool


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"agent config must contain a {name} mapping")
    return value


def load_agent_config(path: str | Path) -> AgentConfig:
    """Load rule thresholds from ``configs/agent.yaml``."""

    config = load_yaml(path)
    section = _mapping(config.get("agent"), "agent")
    thresholds = _mapping(section.get("thresholds"), "agent.thresholds")
    positive_names = (
        "feedback_cooldown_s",
        "tracking_error_high_m",
        "tracking_error_good_m",
        "interaction_force_high_n",
        "task_speed_fast",
        "task_speed_slow",
        "human_power_low_w",
        "inactivity_duration_s",
    )
    values = {
        name: float(section[name] if name in section else thresholds[name])
        for name in positive_names
    }
    values.update(
        {
            "fatigue_high": float(thresholds["fatigue_high"]),
        }
    )
    if any(value < 0.0 or not np.isfinite(value) for value in values.values()):
        raise ValueError("Agent thresholds must be finite and non-negative")
    if values["tracking_error_good_m"] > values["tracking_error_high_m"]:
        raise ValueError("good tracking threshold cannot exceed high-error threshold")
    if values["task_speed_slow"] > values["task_speed_fast"]:
        raise ValueError("slow speed threshold cannot exceed fast speed threshold")
    if not 0.0 <= values["fatigue_high"] <= 1.0:
        raise ValueError("fatigue_high must be in [0,1]")
    return AgentConfig(
        enabled=bool(section.get("enabled", True)),
        mode=str(section.get("mode", "rules_only")),
        speech_enabled=bool(section.get("speech_enabled", False)),
        feedback_cooldown_s=values["feedback_cooldown_s"],
        tracking_error_high_m=values["tracking_error_high_m"],
        tracking_error_good_m=values["tracking_error_good_m"],
        interaction_force_high_n=values["interaction_force_high_n"],
        task_speed_fast=values["task_speed_fast"],
        task_speed_slow=values["task_speed_slow"],
        human_power_low_w=values["human_power_low_w"],
        inactivity_duration_s=values["inactivity_duration_s"],
        fatigue_high=values["fatigue_high"],
        hardware_validation_required=bool(section.get("hardware_validation_required", True)),
    )


@dataclass(frozen=True)
class AgentObservation:
    """Structured read-only input to the Agent.

    ``tracking_error_norm`` is in m/rad task-space norm, force in N, speed in
    task-space units per second, power in W, and fatigue is normalized [0,1].
    """

    task: str
    elapsed_s: float
    tracking_error_norm: float
    interaction_force_norm: float
    task_speed_norm: float
    human_power_w: float
    fatigue: float
    task_progress: float
    safety_status: str


@dataclass(frozen=True)
class AgentEvent:
    """One user-facing feedback event and its structured context."""

    event: str
    message: str
    severity: Severity
    timestamp_s: float
    context: dict[str, Any]


@dataclass(frozen=True)
class AgentSummary:
    """Template-based training summary generated without an LLM."""

    title: str
    message: str
    highlights: list[str]
    recommendation: str
    event_count: int


SpeechSink = Callable[[str], None]


class RuleBasedAgent:
    """Detect events and provide feedback without touching control state."""

    def __init__(self, config: AgentConfig, speech_sink: SpeechSink | None = None) -> None:
        self.config = config
        self._speech_sink = speech_sink
        self._events: list[AgentEvent] = []
        self._last_event_time: dict[str, float] = {}
        self._inactive_duration_s = 0.0
        self._last_observation_time: float | None = None
        self._task: str | None = None
        self._started = False

    @property
    def events(self) -> list[AgentEvent]:
        """Return a copy of all emitted events for persistence."""

        return list(self._events)

    def reset(self) -> None:
        """Clear event history and temporal detectors."""

        self._events.clear()
        self._last_event_time.clear()
        self._inactive_duration_s = 0.0
        self._last_observation_time = None
        self._task = None
        self._started = False

    def _emit(
        self,
        event: str,
        message: str,
        observation: AgentObservation | None,
        severity: Severity = "info",
        force: bool = False,
    ) -> AgentEvent | None:
        if not self.config.enabled:
            return None
        timestamp_s = 0.0 if observation is None else float(observation.elapsed_s)
        previous = self._last_event_time.get(event)
        if (
            not force
            and previous is not None
            and timestamp_s - previous < self.config.feedback_cooldown_s
        ):
            return None
        context: dict[str, Any] = {}
        if observation is not None:
            context = {
                "task": observation.task,
                "duration": observation.elapsed_s,
                "tracking_error": observation.tracking_error_norm,
                "force_level": "high"
                if observation.interaction_force_norm >= self.config.interaction_force_high_n
                else "normal",
                "fatigue_level": "high"
                if observation.fatigue >= self.config.fatigue_high
                else "normal",
            }
        result = AgentEvent(event, message, severity, timestamp_s, context)
        self._last_event_time[event] = timestamp_s
        self._events.append(result)
        LOGGER.info(
            "agent_event event=%s severity=%s message=%s context=%s",
            result.event,
            result.severity,
            result.message,
            result.context,
        )
        if self.config.speech_enabled and self._speech_sink is not None:
            try:
                self._speech_sink(message)
            except Exception:  # noqa: BLE001 - speech failure cannot affect control
                LOGGER.exception("agent_speech_failed")
        return result

    def start(self, task: str, patient_profile: str = "moderate") -> AgentEvent | None:
        """Emit the start prompt for a new session."""

        self.reset()
        self._task = task
        self._started = True
        return self._emit(
            "task_started",
            f"本次{task}训练开始，请保持自然、主动的动作。",
            None,
            severity="info",
            force=True,
        )

    def observe(
        self, observation: AgentObservation, dt_s: float | None = None
    ) -> AgentEvent | None:
        """Evaluate one telemetry sample and emit at most one prioritized event."""

        values = np.asarray(
            [
                observation.elapsed_s,
                observation.tracking_error_norm,
                observation.interaction_force_norm,
                observation.task_speed_norm,
                observation.human_power_w,
                observation.fatigue,
                observation.task_progress,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            return self._emit(
                "agent_input_invalid",
                "训练数据暂时不可用，请保持当前状态。",
                observation,
                "warning",
            )
        if dt_s is None:
            dt_s = (
                0.0
                if self._last_observation_time is None
                else max(0.0, observation.elapsed_s - self._last_observation_time)
            )
        self._last_observation_time = observation.elapsed_s
        if observation.human_power_w < self.config.human_power_low_w:
            self._inactive_duration_s += dt_s
        else:
            self._inactive_duration_s = 0.0

        if observation.safety_status == "fallback":
            return self._emit(
                "safety_stop",
                "安全保护已介入，请暂停并等待治疗师确认。",
                observation,
                "critical",
            )
        if observation.interaction_force_norm >= self.config.interaction_force_high_n:
            return self._emit(
                "force_too_high",
                "请稍微放松握把，当前交互力偏大。",
                observation,
                "warning",
            )
        if observation.fatigue >= self.config.fatigue_high:
            return self._emit(
                "fatigue_detected",
                "检测到连续疲劳趋势，建议暂停休息。",
                observation,
                "warning",
            )
        if observation.tracking_error_norm >= self.config.tracking_error_high_m:
            return self._emit(
                "tracking_error_high",
                "当前轨迹偏差较大，请放慢动作并回到参考轨迹。",
                observation,
                "warning",
            )
        if self._inactive_duration_s >= self.config.inactivity_duration_s:
            return self._emit(
                "patient_inactive",
                "请继续主动完成动作，机器人辅助会保持在当前水平。",
                observation,
                "warning",
            )
        if observation.task_speed_norm > self.config.task_speed_fast:
            return self._emit(
                "speed_too_fast", "当前速度偏快，请放慢并保持平稳。", observation, "warning"
            )
        if 0.0 < observation.task_speed_norm < self.config.task_speed_slow:
            return self._emit(
                "speed_too_slow", "当前速度偏慢，可以轻柔地继续向目标移动。", observation, "info"
            )
        if observation.tracking_error_norm <= self.config.tracking_error_good_m:
            return self._emit(
                "tracking_good", "保持当前节奏，轨迹控制得很好。", observation, "positive"
            )
        return None

    def complete(
        self,
        observation: AgentObservation,
        report: Mapping[str, Any],
    ) -> tuple[AgentEvent | None, AgentSummary]:
        """Emit completion feedback and construct a structured summary."""

        event = self._emit(
            "task_completed", "本次训练完成，做得很好。", observation, "positive", force=True
        )
        completed = bool(report.get("completed", False))
        average_error = float(report.get("average_tracking_error", 0.0))
        peak_force = float(report.get("peak_interaction_force", 0.0))
        active_work = float(report.get("patient_active_work", 0.0))
        if completed:
            message = "训练已完成，建议根据今天的表现安排下一次训练强度。"
        else:
            message = "训练已结束，建议先休息并与治疗师一起查看本次数据。"
        highlights = [
            f"平均轨迹误差 {average_error:.4f}",
            f"峰值交互力 {peak_force:.4f} N",
            f"患者主动做功 {active_work:.4f} J",
        ]
        recommendation = (
            "继续保持主动参与" if active_work > 0.0 else "下次训练可在舒适范围内增加主动动作"
        )
        summary = AgentSummary(
            title="训练总结",
            message=message,
            highlights=highlights,
            recommendation=recommendation,
            event_count=len(self._events),
        )
        return event, summary
