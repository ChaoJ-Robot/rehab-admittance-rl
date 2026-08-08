"""Simulation training session state machine for the Phase 7 page."""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from dotenv import load_dotenv
from rehab_sim.agent import (
    AgentEvent,
    AgentObservation,
    LLMAgent,
    RuleBasedAgent,
    load_agent_config,
    load_llm_config,
)
from rehab_sim.config import load_yaml
from rehab_sim.controllers import AdmittanceParameters
from rehab_sim.safety import (
    SafetyObservation,
    SafetySupervisor,
    load_safety_configuration,
)
from rehab_sim.safety.parameter_projector import parameter_vector
from rehab_sim.tasks import (
    CircleTrajectory,
    FigureEightTrajectory,
    PointToPointTrajectory,
    ReferenceTrajectory,
)

from backend.app.schemas.models import (
    AgentChatMessage,
    AgentEventPayload,
    AgentSummaryPayload,
    ConfigSummary,
    ControlMode,
    PatientProfile,
    SessionSnapshot,
    SessionState,
    StartRequest,
    TaskName,
    Telemetry,
    TrainingReport,
)

LOGGER = logging.getLogger("rehab.backend.session")


@dataclass(frozen=True)
class _LLMTask:
    """One asynchronous LLM job; the control path never awaits it."""

    kind: Literal["enrich", "summary"]
    event: AgentEvent | None = None
    report: dict[str, Any] | None = None
    events: list[AgentEvent] | None = None


COLOR_BLOCK_NAMES: tuple[str, ...] = ("red", "blue", "green", "yellow")
COLOR_BLOCK_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.48, 0.14, 0.0),
    (0.48, -0.14, 0.0),
    (0.24, 0.14, 0.0),
    (0.24, -0.14, 0.0),
)


class _PathFollow:
    """Piecewise-linear waypoint path traversed at constant speed (maze task)."""

    def __init__(self, waypoints: Sequence[Sequence[float]], speed: float) -> None:
        points = np.asarray(waypoints, dtype=np.float64)
        self._points = points
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self._cumulative = np.concatenate([[0.0], np.cumsum(segments)])
        self._total = float(self._cumulative[-1])
        self._speed = float(speed)

    def reference(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        distance = min(self._speed * max(time_s, 0.0), self._total)
        index = int(
            np.clip(
                np.searchsorted(self._cumulative, distance, side="right") - 1,
                0,
                len(self._points) - 2,
            )
        )
        segment_length = float(self._cumulative[index + 1] - self._cumulative[index])
        fraction = (distance - float(self._cumulative[index])) / max(segment_length, 1e-9)
        position = self._points[index] + fraction * (self._points[index + 1] - self._points[index])
        direction = (self._points[index + 1] - self._points[index]) / max(segment_length, 1e-9)
        velocity = (
            direction * self._speed if distance < self._total else np.zeros(3, dtype=np.float64)
        )
        return position, velocity

    def progress(self, time_s: float) -> float:
        return float(min(1.0, self._speed * max(time_s, 0.0) / max(self._total, 1e-6)))


class _ColorMemoryPlan:
    """Memorise-then-recall sequence task between four coloured targets."""

    def __init__(self, sequence: Sequence[int], memorize_s: float, recall_s: float) -> None:
        self.sequence = list(sequence)
        self.memorize_s = float(memorize_s)
        self.recall_s = max(float(recall_s), 0.5)
        self._start = np.asarray([0.36, 0.0, 0.0], dtype=np.float64)
        self._targets = [
            np.asarray(COLOR_BLOCK_POSITIONS[index], dtype=np.float64) for index in self.sequence
        ]

    def phase(self, time_s: float) -> Literal["memorize", "recall"]:
        return "memorize" if time_s < self.memorize_s else "recall"

    def reference(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        if time_s < self.memorize_s or not self._targets:
            return self._start.copy(), np.zeros(3, dtype=np.float64)
        leg_s = self.recall_s / len(self._targets)
        progressed = min(time_s - self.memorize_s, self.recall_s)
        index = min(int(progressed / leg_s), len(self._targets) - 1)
        origin = self._start if index == 0 else self._targets[index - 1]
        fraction = min((progressed - index * leg_s) / leg_s, 1.0)
        target = self._targets[index]
        velocity = (
            (target - origin) / leg_s if fraction < 1.0 else np.zeros(3, dtype=np.float64)
        )
        return origin + fraction * (target - origin), velocity

    def progress(self, time_s: float) -> float:
        if time_s < self.memorize_s:
            return float(0.2 * time_s / max(self.memorize_s, 1e-6))
        recall_fraction = min((time_s - self.memorize_s) / self.recall_s, 1.0)
        return float(0.2 + 0.8 * recall_fraction)


class TrainingSession:
    """Own one simulation-only session and expose a page-safe state machine.

    The session emits task-space telemetry at 20 Hz. It intentionally uses a
    deterministic simulation data source; the Phase 8 Agent only observes this
    stream and cannot modify parameters, commands, or the control path.
    """

    refresh_hz = 20
    _tick_dt_s = 1.0 / refresh_hz

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or Path(__file__).resolve().parents[3])
        load_dotenv(self.root / ".env")
        self._task_config = load_yaml(self.root / "configs" / "tasks.yaml")
        self._admittance_config = load_yaml(self.root / "configs" / "admittance.yaml")
        self._safety_config = load_yaml(self.root / "configs" / "safety.yaml")
        agent_config = load_agent_config(self.root / "configs" / "agent.yaml")
        self._agent = RuleBasedAgent(agent_config)
        self._llm_agent = LLMAgent(load_llm_config(self.root / "configs" / "agent.yaml"))
        self._llm_queue: asyncio.Queue[_LLMTask] = asyncio.Queue()
        self._llm_worker_task: asyncio.Task[None] | None = None
        self._agent_chat: list[AgentChatMessage] = []
        safety = load_safety_configuration(self._admittance_config, self._safety_config)
        self._supervisor = SafetySupervisor(safety)
        self._baseline = parameter_vector(AdmittanceParameters.from_config(self._admittance_config))
        self._session_id = "none"
        self._state: SessionState = "idle"
        self._task_name: TaskName = "point_to_point"
        self._patient_profile: PatientProfile = "moderate"
        self._mode: ControlMode = "fixed"
        self._duration_s = 4.0
        self._elapsed_s = 0.0
        self._progress = 0.0
        self._score = 0.0
        self._parameters = self._baseline.copy()
        self._task: ReferenceTrajectory | None = None
        self._maze: _PathFollow | None = None
        self._maze_walls: list[list[float]] = []
        self._color: _ColorMemoryPlan | None = None
        self._telemetry: Telemetry | None = None
        self._report: TrainingReport | None = None
        self._last_agent_event: AgentEvent | None = None
        self._agent_summary: AgentSummaryPayload | None = None
        self._history: deque[Telemetry] = deque(maxlen=2400)
        self._run_task: asyncio.Task[None] | None = None
        self._last_wall_time = time.monotonic()
        self._error_sum = 0.0
        self._force_sum = 0.0
        self._peak_force = 0.0
        self._human_work = 0.0
        self._robot_work = 0.0
        self._parameter_change = 0.0
        self._smoothness_sum = 0.0
        self._safety_triggers = 0
        self._last_velocity: np.ndarray = np.zeros(3, dtype=np.float64)

    @property
    def history(self) -> list[Telemetry]:
        """Return a copy of buffered telemetry for report consumers."""

        return list(self._history)

    def agent_events(self) -> list[AgentEventPayload]:
        """Return the structured Agent event log for the current session."""

        return [
            AgentEventPayload(
                event=event.event,
                message=event.message,
                severity=event.severity,
                timestamp_s=event.timestamp_s,
                context=event.context,
            )
            for event in self._agent.events
        ]

    def config_summary(self) -> ConfigSummary:
        """Return safe UI configuration choices and simulation limits."""

        safety = self._supervisor.configuration
        return ConfigSummary(
            tasks=[
                "point_to_point",
                "circle_tracking",
                "figure8_tracking",
                "maze_navigation",
                "color_memory",
            ],
            patient_profiles=["mild", "moderate", "severe"],
            modes=["fixed", "rl"],
            refresh_hz=self.refresh_hz,
            simulation_only=True,
            hardware_validation_required=safety.hardware_validation_required,
            parameter_bounds={
                name: [float(lower), float(upper)]
                for name, lower, upper in zip(
                    ("damping_x", "damping_y", "damping_theta", "assist_gain", "velocity_scale"),
                    safety.parameter_lower,
                    safety.parameter_upper,
                    strict=True,
                )
            },
            interaction_force_limit=safety.interaction_force_limit,
            task_speed_limit=safety.task_speed_limit,
        )

    def _build_task(self, task_name: TaskName, duration_s: float) -> ReferenceTrajectory:
        raw = self._task_config["tasks"][task_name]
        start = np.array([0.35, 0.0, 0.0], dtype=np.float64)
        if task_name == "point_to_point":
            return PointToPointTrajectory(
                start,
                float(raw["target_distance"]),
                duration_s,
                float(raw["target_radius"]),
            )
        if task_name == "circle_tracking":
            return CircleTrajectory(
                start,
                float(raw["path_radius"]),
                float(raw["reference_speed"]),
                duration_s,
                float(raw["path_width"]),
            )
        return FigureEightTrajectory(
            start,
            float(raw["path_width"]),
            float(raw["reference_speed"]),
            duration_s,
            float(raw["path_width_tolerance"]),
        )

    def _build_maze(self, raw: Mapping[str, Any]) -> tuple[_PathFollow, list[list[float]]]:
        """Build the S-shaped corridor maze and its waypoint reference path."""

        walls = [
            [0.15, -0.22, 0.60, -0.22],
            [0.15, 0.22, 0.60, 0.22],
            [0.15, -0.22, 0.15, 0.22],
            [0.60, -0.22, 0.60, 0.22],
            [0.28, -0.22, 0.28, 0.10],
            [0.42, -0.10, 0.42, 0.22],
        ]
        waypoints = [
            [0.18, -0.12, 0.0],
            [0.22, 0.15, 0.0],
            [0.32, 0.15, 0.0],
            [0.38, -0.15, 0.0],
            [0.48, -0.15, 0.0],
            [0.55, -0.05, 0.0],
            [0.55, 0.12, 0.0],
        ]
        return _PathFollow(waypoints, float(raw["reference_speed"])), walls

    def _build_color_memory(self, raw: Mapping[str, Any]) -> _ColorMemoryPlan:
        """Build a per-session random colour-sequence memory task."""

        rng = random.Random(self._session_id)
        sequence = [
            rng.randrange(len(COLOR_BLOCK_NAMES)) for _ in range(int(raw["sequence_length"]))
        ]
        memorize_s = float(raw["memorize_duration"])
        return _ColorMemoryPlan(sequence, memorize_s, self._duration_s - memorize_s)

    def _cancel_run_task(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
        self._run_task = None

    async def start(self, request: StartRequest) -> SessionSnapshot:
        """Start a new session and reset all history and metrics."""

        self._cancel_run_task()
        raw_duration = self._task_config["tasks"][request.task]["task_duration"]
        self._duration_s = float(request.duration_s or raw_duration)
        self._session_id = uuid.uuid4().hex
        self._state = "running"
        self._task_name = request.task
        self._patient_profile = request.patient_profile
        self._mode = request.mode
        self._elapsed_s = 0.0
        self._progress = 0.0
        self._score = 0.0
        self._parameters = self._baseline.copy()
        self._task = None
        self._maze = None
        self._maze_walls = []
        self._color = None
        if self._task_name == "maze_navigation":
            self._maze, self._maze_walls = self._build_maze(
                self._task_config["tasks"][self._task_name]
            )
        elif self._task_name == "color_memory":
            self._color = self._build_color_memory(
                self._task_config["tasks"][self._task_name]
            )
        else:
            self._task = self._build_task(self._task_name, self._duration_s)
        self._telemetry = None
        self._report = None
        self._agent_chat.clear()
        try:
            self._last_agent_event = self._agent.start(self._task_name, self._patient_profile)
            if self._last_agent_event is not None:
                self._append_chat("agent", self._last_agent_event.message, "rules")
        except Exception:  # noqa: BLE001 - Agent failure cannot block session start
            LOGGER.exception("agent_start_failed")
            self._last_agent_event = None
        self._agent_summary = None
        self._ensure_llm_worker()
        self._history.clear()
        self._error_sum = 0.0
        self._force_sum = 0.0
        self._peak_force = 0.0
        self._human_work = 0.0
        self._robot_work = 0.0
        self._parameter_change = 0.0
        self._smoothness_sum = 0.0
        self._safety_triggers = 0
        self._last_velocity = np.zeros(3, dtype=np.float64)
        self._last_wall_time = time.monotonic()
        self._run_task = asyncio.create_task(self._run_loop())
        return self.snapshot()

    async def _run_loop(self) -> None:
        try:
            while self._state in ("running", "paused"):
                if self._state == "running":
                    self._tick()
                await asyncio.sleep(self._tick_dt_s)
        except asyncio.CancelledError:
            raise

    def _tick(self) -> None:
        task_phase: Literal["memorize", "recall"] | None = None
        if self._task is not None:
            reference, reference_velocity = self._task.reference(self._elapsed_s)
            progress_target = self._task.progress(self._elapsed_s)
        elif self._maze is not None:
            reference, reference_velocity = self._maze.reference(self._elapsed_s)
            progress_target = self._maze.progress(self._elapsed_s)
        elif self._color is not None:
            reference, reference_velocity = self._color.reference(self._elapsed_s)
            progress_target = self._color.progress(self._elapsed_s)
            task_phase = self._color.phase(self._elapsed_s)
        else:
            return
        phase = 2.0 * np.pi * self._elapsed_s / max(self._duration_s, self._tick_dt_s)
        lag = np.asarray(
            [0.012 * np.sin(phase), 0.008 * np.sin(phase + 0.7), 0.025 * np.sin(phase)],
            dtype=np.float64,
        )
        actual = reference - lag
        velocity = reference_velocity * 0.92
        tracking_error = reference - actual
        tracking_error[2] = float(np.arctan2(np.sin(tracking_error[2]), np.cos(tracking_error[2])))
        interaction_force = np.asarray(
            [
                0.25 * np.sin(phase + 0.3),
                0.18 * np.cos(phase),
                0.04 * np.sin(phase * 0.5),
            ],
            dtype=np.float64,
        )
        if self._mode == "rl":
            raw_action = np.clip(
                np.asarray(
                    [
                        -tracking_error[0] * 8.0,
                        -tracking_error[2] * 2.0,
                        np.linalg.norm(tracking_error[:2]) * 4.0,
                        -np.linalg.norm(velocity) * 0.5,
                    ],
                    dtype=np.float64,
                ),
                -1.0,
                1.0,
            )
        else:
            raw_action = np.zeros(4, dtype=np.float64)
        safety_observation = SafetyObservation(
            interaction_wrench=interaction_force,
            task_velocity=velocity,
            task_acceleration=(velocity - self._last_velocity) / self._tick_dt_s,
        )
        decision = self._supervisor.supervise(
            raw_action,
            self._parameters,
            safety_observation,
            inference_elapsed_s=0.0,
            model_loaded=True,
        )
        previous_parameters = self._parameters.copy()
        self._parameters = decision.parameters.copy()
        if decision.fallback:
            self._safety_triggers += 1
        force_norm = float(np.linalg.norm(interaction_force[:2]))
        error_norm = float(np.linalg.norm(tracking_error))
        progress = progress_target
        progress_delta = max(0.0, progress - self._progress)
        human_power = max(0.0, float(np.dot(interaction_force, velocity)))
        assistance_work = abs(float(self._parameters[3] * np.dot(tracking_error, velocity)))
        jerk = float(np.linalg.norm((velocity - self._last_velocity) / self._tick_dt_s))
        self._score += 2.0 * progress_delta - 1.5 * error_norm - 0.1 * force_norm
        self._error_sum += error_norm
        self._force_sum += force_norm
        self._peak_force = max(self._peak_force, force_norm)
        self._human_work += human_power * self._tick_dt_s
        self._robot_work += assistance_work * self._tick_dt_s
        self._parameter_change += float(np.linalg.norm(self._parameters - previous_parameters))
        self._smoothness_sum += jerk
        self._last_velocity = velocity.copy()
        self._progress = progress
        self._elapsed_s = min(self._duration_s, self._elapsed_s + self._tick_dt_s)
        if self._elapsed_s >= self._duration_s:
            self._state = "completed"
        agent_observation = AgentObservation(
            task=self._task_name,
            elapsed_s=self._elapsed_s,
            tracking_error_norm=error_norm,
            interaction_force_norm=force_norm,
            task_speed_norm=float(np.linalg.norm(velocity)),
            human_power_w=human_power,
            fatigue=float(np.clip(self._elapsed_s / max(self._duration_s, 1.0), 0.0, 1.0)),
            task_progress=self._progress,
            safety_status="fallback" if decision.fallback else "safe",
        )
        agent_event: AgentEvent | None = None
        try:
            agent_event = self._agent.observe(agent_observation, self._tick_dt_s)
        except Exception:  # noqa: BLE001 - Agent failure cannot affect control
            LOGGER.exception("agent_observe_failed")
        if agent_event is not None:
            self._last_agent_event = agent_event
            self._append_chat("agent", agent_event.message, "rules")
            if self._llm_agent.event_enrichment_enabled:
                self._llm_queue.put_nowait(_LLMTask(kind="enrich", event=agent_event))
        completion_event: AgentEvent | None = None
        if self._state == "completed":
            self._report = self._make_report()
            completion_event = self._complete_agent(agent_observation)
        sample_agent_event = completion_event or agent_event
        telemetry = Telemetry(
            timestamp=time.time(),
            elapsed_s=self._elapsed_s,
            task=self._task_name,
            patient_profile=self._patient_profile,
            mode=self._mode,
            state=self._state,
            reference_pose=reference.tolist(),
            actual_pose=actual.tolist(),
            tracking_error=tracking_error.tolist(),
            end_effector_velocity=velocity.tolist(),
            interaction_force=interaction_force.tolist(),
            human_power_w=human_power,
            fatigue=float(np.clip(self._elapsed_s / max(self._duration_s, 1.0), 0.0, 1.0)),
            admittance_parameters=self._parameters.tolist(),
            rl_action=decision.action.tolist(),
            task_progress=self._progress,
            score=self._score,
            safety_status="fallback" if decision.fallback else "safe",
            safety_reasons=list(decision.reasons),
            maze_walls=self._maze_walls,
            color_block_positions=(
                [list(position) for position in COLOR_BLOCK_POSITIONS]
                if self._color is not None
                else []
            ),
            color_block_names=list(COLOR_BLOCK_NAMES) if self._color is not None else [],
            color_sequence=(
                [COLOR_BLOCK_NAMES[index] for index in self._color.sequence]
                if self._color is not None
                else []
            ),
            task_phase=task_phase,
            agent_event=(
                AgentEventPayload(
                    event=sample_agent_event.event,
                    message=sample_agent_event.message,
                    severity=sample_agent_event.severity,
                    timestamp_s=sample_agent_event.timestamp_s,
                    context=sample_agent_event.context,
                )
                if sample_agent_event is not None
                else None
            ),
        )
        self._telemetry = telemetry
        self._history.append(telemetry)

    def pause(self) -> SessionSnapshot:
        """Pause a running session without losing its telemetry history."""

        if self._state != "running":
            raise ValueError("only a running session can be paused")
        self._state = "paused"
        return self.snapshot()

    def resume(self) -> SessionSnapshot:
        """Resume a paused session."""

        if self._state != "paused":
            raise ValueError("only a paused session can be resumed")
        self._state = "running"
        return self.snapshot()

    def stop(self) -> SessionSnapshot:
        """Stop a session and create a partial report."""

        if self._state in ("running", "paused"):
            self._state = "stopped"
            self._report = self._make_report()
            observation = (
                self._agent_observation_from_telemetry()
                if self._telemetry is not None
                else AgentObservation(
                    task=self._task_name,
                    elapsed_s=self._elapsed_s,
                    tracking_error_norm=0.0,
                    interaction_force_norm=0.0,
                    task_speed_norm=0.0,
                    human_power_w=0.0,
                    fatigue=0.0,
                    task_progress=self._progress,
                    safety_status="safe",
                )
            )
            completion_event = self._complete_agent(observation)
            if completion_event is not None:
                self._last_agent_event = completion_event
        self._cancel_run_task()
        return self.snapshot()

    def set_mode(self, mode: ControlMode) -> SessionSnapshot:
        """Switch fixed/RL simulation mode without touching hardware."""

        self._mode = mode
        return self.snapshot()

    def _make_report(self) -> TrainingReport:
        samples = max(1, len(self._history))
        return TrainingReport(
            completed=self._state == "completed",
            completion_rate=1.0 if self._state == "completed" else self._progress,
            duration_s=self._elapsed_s,
            average_tracking_error=self._error_sum / samples,
            peak_interaction_force=self._peak_force,
            mean_interaction_force=self._force_sum / samples,
            motion_smoothness=self._smoothness_sum / samples,
            patient_active_work=self._human_work,
            robot_assistance_work=self._robot_work,
            parameter_change_total=self._parameter_change,
            safety_trigger_count=self._safety_triggers,
            final_score=self._score,
        )

    def _agent_observation_from_telemetry(self) -> AgentObservation:
        """Convert the latest sample to a read-only Agent input for stop."""

        if self._telemetry is None:
            raise RuntimeError("telemetry is not available")
        return AgentObservation(
            task=self._telemetry.task,
            elapsed_s=self._telemetry.elapsed_s,
            tracking_error_norm=float(np.linalg.norm(self._telemetry.tracking_error)),
            interaction_force_norm=float(np.linalg.norm(self._telemetry.interaction_force[:2])),
            task_speed_norm=float(np.linalg.norm(self._telemetry.end_effector_velocity)),
            human_power_w=self._telemetry.human_power_w,
            fatigue=self._telemetry.fatigue,
            task_progress=self._telemetry.task_progress,
            safety_status=self._telemetry.safety_status,
        )

    def _complete_agent(self, observation: AgentObservation) -> AgentEvent | None:
        """Build the rule-based summary and hand a copy to the LLM worker."""

        if self._report is None:
            return None
        try:
            report_data = (
                self._report.model_dump()
                if hasattr(self._report, "model_dump")
                else self._report.dict()
            )
            event, summary = self._agent.complete(observation, report_data)
            self._agent_summary = AgentSummaryPayload(
                title=summary.title,
                message=summary.message,
                highlights=summary.highlights,
                recommendation=summary.recommendation,
                event_count=summary.event_count,
                source="rules",
            )
            if event is not None:
                self._append_chat("agent", event.message, "rules")
            if self._llm_agent.summary_enabled:
                self._llm_queue.put_nowait(
                    _LLMTask(
                        kind="summary",
                        report=report_data,
                        events=list(self._agent.events),
                    )
                )
            return event
        except Exception:  # noqa: BLE001 - Agent failure cannot affect control
            LOGGER.exception("agent_complete_failed")
            return None

    def _append_chat(
        self,
        role: Literal["user", "agent"],
        message: str,
        source: Literal["rules", "llm", "user"],
    ) -> None:
        """Append one message to the interaction feed shown in the UI."""

        self._agent_chat.append(
            AgentChatMessage(role=role, message=message, source=source, timestamp_s=time.time())
        )

    def _ensure_llm_worker(self) -> None:
        """Start the background LLM consumer exactly once per process."""

        if self._llm_worker_task is None or self._llm_worker_task.done():
            self._llm_worker_task = asyncio.create_task(self._llm_worker_loop())

    async def _llm_worker_loop(self) -> None:
        """Drain LLM jobs without ever touching the 20 Hz control loop."""

        while True:
            task = await self._llm_queue.get()
            try:
                await self._process_llm_task(task)
            except Exception:  # noqa: BLE001 - LLM failure must not propagate
                LOGGER.exception("llm_task_failed kind=%s", task.kind)
            finally:
                self._llm_queue.task_done()

    async def _process_llm_task(self, task: _LLMTask) -> None:
        """Handle one queued LLM job and publish its result to the snapshot."""

        if task.kind == "enrich" and task.event is not None:
            message = await self._llm_agent.enrich_event(task.event)
            if message:
                self._append_chat("agent", message, "llm")
            return
        if task.kind == "summary" and task.report is not None and task.events is not None:
            summary = await self._llm_agent.generate_summary(task.report, task.events)
            if summary is not None:
                self._agent_summary = AgentSummaryPayload(
                    title=summary.title,
                    message=summary.message,
                    highlights=summary.highlights,
                    recommendation=summary.recommendation,
                    event_count=summary.event_count,
                    source="llm",
                )
                self._append_chat("agent", summary.message, "llm")

    async def chat(self, message: str) -> str | None:
        """Answer a patient/therapist question, or None when the LLM is off."""

        if not self._llm_agent.chat_enabled:
            return None
        self._ensure_llm_worker()
        self._append_chat("user", message, "user")
        context: dict[str, Any] = {
            "task": self._task_name,
            "patient_profile": self._patient_profile,
            "mode": self._mode,
            "elapsed_s": round(self._elapsed_s, 2),
            "task_progress": round(self._progress, 3),
            "score": round(self._score, 3),
        }
        if self._report is not None:
            context.update(self._report.model_dump())
        elif self._telemetry is not None:
            telemetry = self._telemetry
            context.update(
                {
                    "tracking_error": round(float(np.linalg.norm(telemetry.tracking_error)), 4),
                    "interaction_force": round(
                        float(np.linalg.norm(telemetry.interaction_force[:2])), 4
                    ),
                    "human_power_w": round(telemetry.human_power_w, 4),
                    "fatigue": round(telemetry.fatigue, 3),
                    "safety_status": telemetry.safety_status,
                }
            )
        history = [
            {
                "role": "user" if item.role == "user" else "assistant",
                "content": item.message,
            }
            for item in self._agent_chat[-10:]
            if item.source in ("user", "llm")
        ]
        reply = await self._llm_agent.answer(message, context, history)
        if reply is None:
            LOGGER.warning("llm_chat_unavailable")
            return None
        self._append_chat("agent", reply, "llm")
        return reply

    def snapshot(self) -> SessionSnapshot:
        """Build a serializable snapshot without exposing mutable internals."""

        return SessionSnapshot(
            session_id=self._session_id,
            state=self._state,
            task=self._task_name,
            patient_profile=self._patient_profile,
            mode=self._mode,
            elapsed_s=self._elapsed_s,
            duration_s=self._duration_s,
            task_progress=self._progress,
            score=self._score,
            telemetry=self._telemetry,
            report=self._report,
            agent_event=(
                AgentEventPayload(
                    event=self._last_agent_event.event,
                    message=self._last_agent_event.message,
                    severity=self._last_agent_event.severity,
                    timestamp_s=self._last_agent_event.timestamp_s,
                    context=self._last_agent_event.context,
                )
                if self._last_agent_event is not None
                else None
            ),
            agent_summary=self._agent_summary,
            agent_chat=list(self._agent_chat),
        )
