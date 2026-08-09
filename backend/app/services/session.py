"""Simulation training session state machine for the Phase 7 page."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from dotenv import load_dotenv
from rehab_sim.agent import (
    AgentEvent,
    AgentObservation,
    LLMAgent,
    RuleBasedAgent,
    assess_history,
    load_agent_config,
    load_llm_config,
    recommend_next_session,
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
    generate_grid_maze,
)

from backend.app.schemas.models import (
    AgentChatMessage,
    AgentEventPayload,
    AgentSummaryPayload,
    AssignmentPayload,
    ConfigSummary,
    ControlMode,
    PatientAssessmentPayload,
    PatientClinicalProfilePayload,
    PatientHistoryEntryPayload,
    PatientProfile,
    PatientSummary,
    SessionPrescriptionPayload,
    SessionSnapshot,
    SessionState,
    StartRequest,
    TaskName,
    TaskParamSpec,
    Telemetry,
    TrainingCheckInPayload,
    TrainingReport,
)
from backend.app.services.patient_store import (
    PatientClinicalProfile,
    PatientHistoryEntry,
    PatientStore,
    TaskAssignment,
    validate_patient_id,
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


def _point_segment_distance(point: np.ndarray, wall: Sequence[float]) -> float:
    """Shortest planar distance from a task-space point to one wall segment."""

    begin = np.asarray(wall[:2], dtype=np.float64)
    end = np.asarray(wall[2:4], dtype=np.float64)
    segment = end - begin
    fraction = float(
        np.clip(
            np.dot(point[:2] - begin, segment) / max(float(np.dot(segment, segment)), 1e-12), 0, 1
        )
    )
    return float(np.linalg.norm(point[:2] - (begin + fraction * segment)))


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

    def actual_progress(self, pose: Sequence[float] | np.ndarray[Any, Any]) -> float:
        """Project an observed pose onto the route and return monotonic path progress."""

        point = np.asarray(pose, dtype=np.float64)
        best_distance = float("inf")
        best_along = 0.0
        for index in range(len(self._points) - 1):
            begin = self._points[index]
            end = self._points[index + 1]
            segment = end - begin
            length_sq = float(np.dot(segment, segment))
            fraction = float(np.clip(np.dot(point - begin, segment) / max(length_sq, 1e-12), 0, 1))
            projected = begin + fraction * segment
            distance = float(np.linalg.norm(point[:2] - projected[:2]))
            if distance < best_distance:
                best_distance = distance
                best_along = float(self._cumulative[index]) + fraction * float(
                    self._cumulative[index + 1] - self._cumulative[index]
                )
        return float(np.clip(best_along / max(self._total, 1e-9), 0.0, 1.0))

    @property
    def total_length(self) -> float:
        return self._total

    @property
    def start(self) -> np.ndarray:
        return self._points[0]

    @property
    def points(self) -> list[list[float]]:
        return self._points.tolist()

    @property
    def goal(self) -> np.ndarray:
        """Final waypoint, shown as the maze goal marker in the UI."""

        return self._points[-1]


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
        velocity = (target - origin) / leg_s if fraction < 1.0 else np.zeros(3, dtype=np.float64)
        return origin + fraction * (target - origin), velocity

    def progress(self, time_s: float) -> float:
        if time_s < self.memorize_s:
            return float(0.2 * time_s / max(self.memorize_s, 1e-6))
        recall_fraction = min((time_s - self.memorize_s) / self.recall_s, 1.0)
        return float(0.2 + 0.8 * recall_fraction)


class _FollowReachPlan:
    """Visit a sequence of targets, holding on each for a dwell interval.

    The reference moves from the start pose to each target in turn and pauses
    ``dwell_s`` on every target so the patient can "arrive" before moving on.
    """

    def __init__(
        self,
        targets: Sequence[Sequence[float] | np.ndarray[Any, Any]],
        speed: float,
        dwell_s: float,
        start_pose: Sequence[float] | np.ndarray[Any, Any],
    ) -> None:
        self._start = np.asarray(start_pose, dtype=np.float64)
        self._targets = [np.asarray(target, dtype=np.float64) for target in targets]
        self._speed = max(float(speed), 1e-3)
        self._dwell_s = max(float(dwell_s), 0.0)
        self._segments = [self._start, *self._targets[:-1]]
        self._leg_distances = [
            float(np.linalg.norm(target - origin))
            for origin, target in zip(self._segments, self._targets, strict=True)
        ]
        self._leg_times = [distance / self._speed for distance in self._leg_distances]
        self._total_distance = sum(self._leg_distances)

    @property
    def targets(self) -> list[list[float]]:
        """Return the ordered target positions for the frontend chart."""

        return [list(target) for target in self._targets]

    def current_target(self, time_s: float) -> list[float]:
        """Return the target currently being approached (visual-guided mode)."""

        _, _, _, index = self._stage(time_s)
        return list(self._targets[index])

    def _stage(self, time_s: float) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Resolve (origin, target, fraction, index) for ``time_s``."""

        elapsed = max(time_s, 0.0)
        for index, (leg_s, target) in enumerate(zip(self._leg_times, self._targets, strict=True)):
            origin = self._segments[index]
            if elapsed < leg_s:
                return origin, target, elapsed / leg_s, index
            elapsed -= leg_s
            if elapsed < self._dwell_s:
                return target, target, 1.0, index
            elapsed -= self._dwell_s
        last = self._targets[-1]
        return last, last, 1.0, len(self._targets) - 1

    def reference(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return reference pose and velocity at ``time_s``."""

        origin, target, fraction, _ = self._stage(time_s)
        pose = origin + fraction * (target - origin)
        direction = target - origin
        distance = float(np.linalg.norm(direction))
        velocity = (
            direction / max(distance, 1e-9) * self._speed
            if 0.0 < fraction < 1.0
            else np.zeros(3, dtype=np.float64)
        )
        return pose, velocity

    def progress(self, time_s: float) -> float:
        """Return path progress; dwell time does not advance progress."""

        if self._total_distance <= 0.0:
            return 1.0
        _, _, fraction, index = self._stage(time_s)
        completed = sum(self._leg_distances[:index])
        return float(
            np.clip(
                (completed + fraction * self._leg_distances[index]) / self._total_distance,
                0.0,
                1.0,
            )
        )


class _InterceptPlan:
    """A moving target shuttling along a straight corridor to be intercepted.

    The reference itself is the moving target: the patient must keep the
    end-effector close to it while it reverses direction at each endpoint.
    """

    def __init__(
        self,
        point_a: Sequence[float],
        point_b: Sequence[float],
        speed: float,
        duration_s: float,
    ) -> None:
        self._a = np.asarray(point_a, dtype=np.float64)
        self._b = np.asarray(point_b, dtype=np.float64)
        self._speed = max(float(speed), 1e-3)
        self._duration_s = max(float(duration_s), 1e-3)
        direction = self._b - self._a
        self._length = float(np.linalg.norm(direction))
        self._unit = direction / max(self._length, 1e-9)

    @property
    def endpoints(self) -> list[list[float]]:
        """Corridor endpoints drawn by the frontend chart."""

        return [list(self._a), list(self._b)]

    def reference(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        elapsed = max(time_s, 0.0)
        if self._length <= 1e-9:
            return self._a.copy(), np.zeros(3, dtype=np.float64)
        phase = (elapsed * self._speed) % (2.0 * self._length)
        distance = phase if phase <= self._length else 2.0 * self._length - phase
        position = self._a + (distance / self._length) * (self._b - self._a)
        heading = 1.0 if phase < self._length else -1.0
        velocity = self._unit * self._speed * heading
        return position, velocity

    def progress(self, time_s: float) -> float:
        """Interception is time-bounded rather than path-bounded."""

        return float(min(1.0, max(time_s, 0.0) / self._duration_s))


class _MarkerMemoryPlan:
    """Spatial memory rounds: memorise a visible marker, then reach it blind.

    Each round shows the marker for ``memorize_s`` while the reference holds
    the start pose, then hides it and moves toward the remembered position
    over ``recall_s``.
    """

    def __init__(
        self,
        markers: Sequence[Sequence[float] | np.ndarray[Any, Any]],
        memorize_s: float,
        recall_s: float,
        start_pose: Sequence[float] | np.ndarray[Any, Any],
    ) -> None:
        self._markers = [np.asarray(marker, dtype=np.float64) for marker in markers]
        self.memorize_s = max(float(memorize_s), 0.2)
        self.recall_s = max(float(recall_s), 0.5)
        self._start = np.asarray(start_pose, dtype=np.float64)
        self._round_s = self.memorize_s + self.recall_s
        self._rounds = max(len(self._markers), 1)

    def _round_index(self, time_s: float) -> int:
        return min(int(max(time_s, 0.0) // self._round_s), self._rounds - 1)

    def phase(self, time_s: float) -> Literal["memorize", "recall"]:
        index = self._round_index(time_s)
        local = max(time_s, 0.0) - index * self._round_s
        return "memorize" if local <= self.memorize_s else "recall"

    def marker(self, time_s: float) -> list[float]:
        return list(self._markers[self._round_index(time_s)])

    def marker_visible(self, time_s: float) -> bool:
        return self.phase(time_s) == "memorize"

    def reference(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        index = self._round_index(time_s)
        local = max(time_s, 0.0) - index * self._round_s
        marker = self._markers[index]
        if local <= self.memorize_s:
            return self._start.copy(), np.zeros(3, dtype=np.float64)
        fraction = min((local - self.memorize_s) / self.recall_s, 1.0)
        pose = self._start + fraction * (marker - self._start)
        velocity = (
            (marker - self._start) / self.recall_s
            if fraction < 1.0
            else np.zeros(3, dtype=np.float64)
        )
        return pose, velocity

    def progress(self, time_s: float) -> float:
        index = self._round_index(time_s)
        local = max(time_s, 0.0) - index * self._round_s
        within = 0.0 if local <= self.memorize_s else (local - self.memorize_s) / self.recall_s
        return float(np.clip((index + min(within, 1.0)) / self._rounds, 0.0, 1.0))


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
        self._llm_queue: asyncio.Queue[_LLMTask] | None = None
        self._llm_loop: asyncio.AbstractEventLoop | None = None
        self._llm_worker_task: asyncio.Task[None] | None = None
        self._agent_chat: list[AgentChatMessage] = []
        self._patient_store = PatientStore(self.root)
        safety = load_safety_configuration(self._admittance_config, self._safety_config)
        self._supervisor = SafetySupervisor(safety)
        self._baseline = parameter_vector(AdmittanceParameters.from_config(self._admittance_config))
        self._session_id = "none"
        self._assignment_id: str | None = None
        self._task_params: dict[str, Any] = {}
        self._check_in = TrainingCheckInPayload()
        self._state: SessionState = "idle"
        self._task_name: TaskName = "point_to_point"
        self._patient_id = "default"
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
        self._follow: _FollowReachPlan | None = None
        self._guided_only = False
        self._intercept: _InterceptPlan | None = None
        self._marker_memory: _MarkerMemoryPlan | None = None
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
        self._last_actual: np.ndarray | None = None
        self._actual_path_length = 0.0
        self._optimal_path_length = 0.0
        self._collision_count = 0
        self._collision_active = False
        self._target_hit_count = 0

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
        task_params: dict[str, list[TaskParamSpec]] = {}
        for task_name, raw in self._task_config["tasks"].items():
            adjustable = raw.get("adjustable", [])
            if isinstance(adjustable, list):
                task_params[task_name] = [TaskParamSpec(**spec) for spec in adjustable]
        return ConfigSummary(
            tasks=list(self._task_config["tasks"].keys()),
            task_params=task_params,
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

    @staticmethod
    def _merged_params(raw: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
        """Merge per-session parameter overrides over the YAML defaults."""

        merged = dict(raw)
        for key, value in overrides.items():
            if key in merged:
                merged[key] = value
        return merged

    def _validated_task_params(
        self, task_name: TaskName, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate UI/assignment overrides against the task's declared controls."""

        raw = self._task_config["tasks"][task_name]
        specs = {
            str(spec["name"]): spec
            for spec in raw.get("adjustable", [])
            if isinstance(spec, Mapping) and "name" in spec
        }
        unknown = sorted(str(key) for key in params if str(key) not in specs)
        if unknown:
            raise ValueError(f"unsupported task parameters for {task_name}: {', '.join(unknown)}")
        validated: dict[str, Any] = {}
        for key, value in params.items():
            name = str(key)
            spec = specs[name]
            if spec.get("type", "slider") == "select":
                options = list(spec.get("options", []))
                match = next((option for option in options if str(option) == str(value)), None)
                if match is None:
                    raise ValueError(f"invalid option for {name}: {value!r}")
                validated[name] = match
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be numeric") from error
            minimum = float(spec["min"]) if spec.get("min") is not None else -float("inf")
            maximum = float(spec["max"]) if spec.get("max") is not None else float("inf")
            if not minimum <= numeric <= maximum:
                raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
            validated[name] = numeric
        return validated

    def _build_task(
        self, task_name: TaskName, duration_s: float, overrides: Mapping[str, Any]
    ) -> ReferenceTrajectory:
        raw = self._merged_params(self._task_config["tasks"][task_name], overrides)
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

    def _build_maze(
        self, raw: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> tuple[_PathFollow, list[list[float]]]:
        """Build the selected maze layout and its waypoint reference path."""

        map_name = str(overrides.get("map") or raw.get("default_map", "s_shape"))
        layouts = raw.get("maps", {})
        speed = float(overrides.get("reference_speed", raw.get("reference_speed", 0.14)))
        procedural = {
            "procedural_easy": (6, 5),
            "procedural_medium": (8, 6),
            "procedural_challenge": (10, 7),
        }
        if map_name in procedural:
            columns, rows = procedural[map_name]
            seed = int(overrides.get("maze_seed", 1))
            generated = generate_grid_maze(columns=columns, rows=rows, seed=seed)
            return _PathFollow(generated.waypoints, speed), generated.walls
        if map_name not in layouts:
            raise ValueError(f"unknown maze map: {map_name}")
        layout = layouts[map_name]
        walls = [[float(value) for value in wall] for wall in layout.get("walls", [])]
        waypoints = [[0.35, 0.0, 0.0]]
        waypoints.extend(
            [[float(value) for value in point] for point in layout.get("waypoints", [])]
        )
        return _PathFollow(waypoints, speed), walls

    def _build_color_memory(
        self, raw: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> _ColorMemoryPlan:
        """Build a per-session random colour-sequence memory task."""

        rng = random.Random(self._session_id)
        sequence_length = int(overrides.get("sequence_length", raw.get("sequence_length", 4)))
        sequence = [rng.randrange(len(COLOR_BLOCK_NAMES)) for _ in range(sequence_length)]
        memorize_s = float(overrides.get("memorize_duration", raw["memorize_duration"]))
        return _ColorMemoryPlan(sequence, memorize_s, self._duration_s - memorize_s)

    def _random_targets(self, rng: random.Random, count: int) -> list[np.ndarray]:
        """Spread ``count`` targets across the workspace, away from home."""

        start = np.array([0.35, 0.0, 0.0], dtype=np.float64)
        targets: list[np.ndarray] = []
        attempts = 0
        while len(targets) < count and attempts < 300:
            attempts += 1
            candidate = np.array(
                [rng.uniform(0.18, 0.55), rng.uniform(-0.18, 0.18), 0.0],
                dtype=np.float64,
            )
            if np.linalg.norm(candidate[:2] - start[:2]) < 0.12:
                continue
            if any(np.linalg.norm(candidate[:2] - target[:2]) < 0.12 for target in targets):
                continue
            targets.append(candidate)
        fallback = [
            [0.45, 0.12],
            [0.45, -0.12],
            [0.24, 0.12],
            [0.24, -0.12],
            [0.35, 0.16],
            [0.35, -0.16],
        ]
        while len(targets) < count:
            anchor = fallback[len(targets) % len(fallback)]
            targets.append(np.array([anchor[0], anchor[1], 0.0], dtype=np.float64))
        return targets

    def _build_follow_reach(
        self, raw: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> _FollowReachPlan:
        """Generate a random target sequence for the follow-to-reach task."""

        rng = random.Random(self._session_id)
        count = int(overrides.get("target_count", raw.get("target_count", 4)))
        speed = float(overrides.get("reference_speed", raw.get("reference_speed", 0.12)))
        dwell_s = float(overrides.get("dwell_time", raw.get("dwell_time", 0.5)))
        start = np.array([0.35, 0.0, 0.0], dtype=np.float64)
        targets = self._random_targets(rng, count)
        return _FollowReachPlan(targets, speed, dwell_s, start)

    def _build_guided_reach(
        self, raw: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> _FollowReachPlan:
        """Sequential reaching where only the current target stays visible."""

        rng = random.Random(self._session_id)
        count = int(overrides.get("target_count", raw.get("target_count", 4)))
        speed = float(overrides.get("reference_speed", raw.get("reference_speed", 0.12)))
        dwell_s = float(raw.get("dwell_time", 0.3))
        start = np.array([0.35, 0.0, 0.0], dtype=np.float64)
        targets = self._random_targets(rng, count)
        return _FollowReachPlan(targets, speed, dwell_s, start)

    def _build_intercept(
        self, raw: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> _InterceptPlan:
        """Random straight corridor whose endpoints shuttle a moving target."""

        rng = random.Random(self._session_id)
        speed = float(overrides.get("reference_speed", raw.get("reference_speed", 0.08)))
        path_length = float(overrides.get("path_length", raw.get("path_length", 0.18)))
        angle = rng.uniform(0.0, 2.0 * math.pi)
        offset = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        offset = offset * (max(path_length, 0.06) / 2.0)
        center = np.array([0.36, 0.0], dtype=np.float64)
        point_a = np.clip(center - offset, [0.18, -0.18], [0.57, 0.18])
        point_b = np.clip(center + offset, [0.18, -0.18], [0.57, 0.18])
        return _InterceptPlan(
            [float(point_a[0]), float(point_a[1]), 0.0],
            [float(point_b[0]), float(point_b[1]), 0.0],
            speed,
            self._duration_s,
        )

    def _build_marker_memory(
        self, raw: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> _MarkerMemoryPlan:
        """Spatial memorise-then-reach rounds spread over the session."""

        rng = random.Random(self._session_id)
        count = int(overrides.get("marker_count", raw.get("marker_count", 3)))
        memorize_s = float(overrides.get("memorize_duration", raw["memorize_duration"]))
        recall_s = max((self._duration_s - count * memorize_s) / max(count, 1), 0.5)
        start = np.array([0.35, 0.0, 0.0], dtype=np.float64)
        markers = self._random_targets(rng, count)
        return _MarkerMemoryPlan(markers, memorize_s, recall_s, start)

    def _cancel_run_task(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
        self._run_task = None

    async def start(self, request: StartRequest) -> SessionSnapshot:
        """Start a new session; a returning patient resumes their parameters."""

        if not validate_patient_id(request.patient_id):
            raise ValueError(f"invalid patient id: {request.patient_id!r}")
        self._cancel_run_task()
        self._patient_id = request.patient_id
        record = self._patient_store.load(self._patient_id)
        raw_config = self._task_config["tasks"][request.task]
        overrides = self._validated_task_params(request.task, request.task_params or {})
        raw_duration = raw_config["task_duration"]
        override_duration = overrides.get("task_duration")
        try:
            override_value = float(override_duration) if override_duration is not None else None
        except (TypeError, ValueError):
            override_value = None
        self._duration_s = float(request.duration_s or override_value or raw_duration)
        self._session_id = uuid.uuid4().hex
        self._assignment_id = request.assignment_id or None
        self._task_params = dict(overrides)
        self._check_in = request.check_in
        self._state = "running"
        self._task_name = request.task
        self._patient_profile = request.patient_profile or cast(
            PatientProfile, record.profile if record is not None else "moderate"
        )
        self._mode = request.mode
        self._elapsed_s = 0.0
        self._progress = 0.0
        self._score = 0.0
        if record is not None and record.latest_parameters:
            # Seamless handover: resume from the parameters the patient
            # ended the previous session with instead of the generic baseline.
            self._parameters = np.asarray(record.latest_parameters, dtype=np.float64)
            safety = self._supervisor.configuration
            self._parameters = np.clip(
                self._parameters,
                np.asarray(safety.parameter_lower, dtype=np.float64),
                np.asarray(safety.parameter_upper, dtype=np.float64),
            )
        else:
            self._parameters = self._baseline.copy()
        self._task = None
        self._maze = None
        self._maze_walls = []
        self._color = None
        self._follow = None
        self._guided_only = False
        self._intercept = None
        self._marker_memory = None
        if self._task_name == "maze_navigation":
            self._maze, self._maze_walls = self._build_maze(raw_config, overrides)
        elif self._task_name == "color_memory":
            self._color = self._build_color_memory(raw_config, overrides)
        elif self._task_name == "follow_to_reach":
            self._follow = self._build_follow_reach(raw_config, overrides)
        elif self._task_name == "visual_guided_reach":
            self._follow = self._build_guided_reach(raw_config, overrides)
            self._guided_only = True
        elif self._task_name == "motion_intercept":
            self._intercept = self._build_intercept(raw_config, overrides)
        elif self._task_name == "marker_memory":
            self._marker_memory = self._build_marker_memory(raw_config, overrides)
        else:
            self._task = self._build_task(self._task_name, self._duration_s, overrides)
        self._telemetry = None
        self._report = None
        self._agent_chat.clear()
        try:
            check_in_data = (
                self._check_in.model_dump()
                if hasattr(self._check_in, "model_dump")
                else self._check_in.dict()
            )
            self._last_agent_event = self._agent.start(
                self._task_name, self._patient_profile, check_in_data
            )
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
        self._last_actual = None
        self._actual_path_length = 0.0
        self._optimal_path_length = (
            float(self._maze.total_length) if self._maze is not None else 0.0
        )
        self._collision_count = 0
        self._collision_active = False
        self._target_hit_count = 0
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
        elif self._follow is not None:
            reference, reference_velocity = self._follow.reference(self._elapsed_s)
            progress_target = self._follow.progress(self._elapsed_s)
        elif self._intercept is not None:
            reference, reference_velocity = self._intercept.reference(self._elapsed_s)
            progress_target = self._intercept.progress(self._elapsed_s)
        elif self._marker_memory is not None:
            reference, reference_velocity = self._marker_memory.reference(self._elapsed_s)
            progress_target = self._marker_memory.progress(self._elapsed_s)
            task_phase = self._marker_memory.phase(self._elapsed_s)
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
        if self._last_actual is not None:
            self._actual_path_length += float(np.linalg.norm(actual[:2] - self._last_actual[:2]))
        self._last_actual = actual.copy()
        if self._maze is not None:
            progress_target = max(self._progress, self._maze.actual_progress(actual))
            self._target_hit_count = max(
                self._target_hit_count,
                int(round(progress_target * max(len(self._maze.points) - 1, 1))),
            )
            near_wall = any(
                _point_segment_distance(actual, wall) <= 0.009 for wall in self._maze_walls
            )
            if near_wall and not self._collision_active:
                self._collision_count += 1
            self._collision_active = near_wall
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
        task_success = False
        if self._maze is not None:
            task_success = (
                self._progress >= 0.985
                and float(np.linalg.norm(actual[:2] - self._maze.goal[:2])) <= 0.03
            )
            if task_success:
                self._state = "completed"
            elif self._elapsed_s >= self._duration_s:
                self._state = "stopped"
        elif self._elapsed_s >= self._duration_s:
            self._state = "completed"
            task_success = True
        baseline_fatigue = float(self._check_in.fatigue_0_10) / 10.0
        fatigue_estimate = float(
            np.clip(
                baseline_fatigue
                + (1.0 - baseline_fatigue) * 0.45 * self._elapsed_s / max(self._duration_s, 1.0),
                0.0,
                1.0,
            )
        )
        agent_observation = AgentObservation(
            task=self._task_name,
            elapsed_s=self._elapsed_s,
            tracking_error_norm=error_norm,
            interaction_force_norm=force_norm,
            task_speed_norm=float(np.linalg.norm(velocity)),
            human_power_w=human_power,
            fatigue=fatigue_estimate,
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
                self._queue_llm_task(_LLMTask(kind="enrich", event=agent_event))
        completion_event: AgentEvent | None = None
        if self._state in ("completed", "stopped"):
            self._report = self._make_report()
            self._persist_patient()
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
            fatigue=fatigue_estimate,
            active_participation_ratio=(
                self._human_work / (self._human_work + self._robot_work)
                if self._human_work + self._robot_work > 1e-9
                else 0.0
            ),
            robot_assistance_ratio=(
                self._robot_work / (self._human_work + self._robot_work)
                if self._human_work + self._robot_work > 1e-9
                else 0.0
            ),
            admittance_parameters=self._parameters.tolist(),
            rl_action=decision.action.tolist(),
            task_progress=self._progress,
            score=self._score,
            safety_status="fallback" if decision.fallback else "safe",
            safety_reasons=list(decision.reasons),
            maze_walls=self._maze_walls,
            maze_start=[0.35, 0.0, 0.0] if self._maze is not None else [],
            maze_goal=(
                [float(value) for value in self._maze.goal] if self._maze is not None else []
            ),
            maze_optimal_path=self._maze.points if self._maze is not None else [],
            maze_collision_count=self._collision_count,
            maze_path_efficiency=(
                float(np.clip(self._optimal_path_length / self._actual_path_length, 0.0, 1.0))
                if self._maze is not None and self._actual_path_length > 1e-9
                else None
            ),
            task_success=task_success,
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
            task_targets=(
                [self._follow.current_target(self._elapsed_s)]
                if self._follow is not None and self._guided_only
                else self._follow.targets
                if self._follow is not None
                else self._intercept.endpoints
                if self._intercept is not None
                else []
            ),
            memory_marker=(
                self._marker_memory.marker(self._elapsed_s)
                if self._marker_memory is not None
                and self._marker_memory.marker_visible(self._elapsed_s)
                else []
            ),
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
            self._persist_patient()
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
        total_work = self._human_work + self._robot_work
        active_ratio = self._human_work / total_work if total_work > 1e-9 else 0.0
        assistance_ratio = self._robot_work / total_work if total_work > 1e-9 else 0.0
        path_efficiency = None
        if self._maze is not None and self._actual_path_length > 1e-9:
            path_efficiency = float(
                np.clip(self._optimal_path_length / self._actual_path_length, 0.0, 1.0)
            )
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
            active_participation_ratio=active_ratio,
            robot_assistance_ratio=assistance_ratio,
            path_efficiency=path_efficiency,
            collision_count=self._collision_count,
            target_hit_count=self._target_hit_count,
        )

    def _persist_patient(self) -> None:
        """Write the finished session into the patient's persistent record."""

        if self._report is None:
            return
        record = self._patient_store.load(self._patient_id)
        if record is None:
            record = self._patient_store.create(self._patient_id, self._patient_profile)
        record.profile = self._patient_profile
        record.latest_parameters = [float(value) for value in self._parameters]
        record.last_session_at = time.time()
        record.session_count += 1
        record.history.append(
            PatientHistoryEntry(
                session_id=self._session_id,
                task=self._task_name,
                timestamp=record.last_session_at,
                duration_s=self._report.duration_s,
                score=self._report.final_score,
                completion_rate=self._report.completion_rate,
                average_tracking_error=self._report.average_tracking_error,
                final_parameters=[float(value) for value in self._parameters],
                mode=self._mode,
                task_params=dict(self._task_params),
                check_in=(
                    self._check_in.model_dump()
                    if hasattr(self._check_in, "model_dump")
                    else self._check_in.dict()
                ),
                peak_interaction_force=self._report.peak_interaction_force,
                active_participation_ratio=self._report.active_participation_ratio,
                robot_assistance_ratio=self._report.robot_assistance_ratio,
                safety_trigger_count=self._report.safety_trigger_count,
                path_efficiency=self._report.path_efficiency,
                collision_count=self._report.collision_count,
                target_hit_count=self._report.target_hit_count,
            )
        )
        if self._assignment_id:
            # Closed loop: finishing a dispatched task settles its assignment.
            for assignment in record.assignments:
                if (
                    assignment.assignment_id == self._assignment_id
                    and assignment.status == "pending"
                ):
                    assignment.status = "completed"
                    assignment.completed_at = record.last_session_at
                    assignment.completed_session = self._session_id
                    break
        try:
            self._patient_store.save(record)
        except OSError as error:
            LOGGER.warning("patient_record_save_failed id=%s error=%s", self._patient_id, error)

    def patient_summary(self, patient_id: str) -> PatientSummary | None:
        """Return one patient's persistent summary, or None when unknown."""

        record = self._patient_store.load(patient_id)
        if record is None:
            return None
        return PatientSummary(
            patient_id=record.patient_id,
            profile=cast(PatientProfile, record.profile),
            created_at=record.created_at,
            last_session_at=record.last_session_at,
            session_count=record.session_count,
            latest_parameters=record.latest_parameters,
            clinical_profile=PatientClinicalProfilePayload(**vars(record.clinical_profile)),
            history=[
                PatientHistoryEntryPayload(
                    session_id=entry.session_id,
                    task=cast(TaskName, entry.task),
                    timestamp=entry.timestamp,
                    duration_s=entry.duration_s,
                    score=entry.score,
                    completion_rate=entry.completion_rate,
                    average_tracking_error=entry.average_tracking_error,
                    final_parameters=entry.final_parameters,
                    mode=cast(ControlMode, entry.mode),
                    task_params=entry.task_params,
                    check_in=TrainingCheckInPayload(**entry.check_in),
                    peak_interaction_force=entry.peak_interaction_force,
                    active_participation_ratio=entry.active_participation_ratio,
                    robot_assistance_ratio=entry.robot_assistance_ratio,
                    safety_trigger_count=entry.safety_trigger_count,
                    path_efficiency=entry.path_efficiency,
                    collision_count=entry.collision_count,
                    target_hit_count=entry.target_hit_count,
                )
                for entry in record.history
            ],
        )

    def register_patient(self, patient_id: str, profile: PatientProfile) -> PatientSummary:
        """Create or update a patient's profile and return their summary."""

        record = self._patient_store.load(patient_id)
        if record is None:
            record = self._patient_store.create(patient_id, profile)
        record.profile = profile
        self._patient_store.save(record)
        summary = self.patient_summary(patient_id)
        if summary is None:
            raise RuntimeError("patient record vanished after save")
        return summary

    def update_clinical_profile(
        self, patient_id: str, payload: Mapping[str, Any]
    ) -> PatientSummary | None:
        """Persist therapist-maintained clinical context for one patient."""

        record = self._patient_store.load(patient_id)
        if record is None:
            return None
        validated = PatientClinicalProfilePayload(**dict(payload))
        record.clinical_profile = PatientClinicalProfile.from_mapping(
            validated.model_dump() if hasattr(validated, "model_dump") else validated.dict()
        )
        self._patient_store.save(record)
        return self.patient_summary(patient_id)

    def list_patients(self) -> list[PatientSummary]:
        """Return summaries for every registered patient."""

        return [
            summary
            for patient_id in self._patient_store.list_ids()
            if (summary := self.patient_summary(patient_id)) is not None
        ]

    def patient_assignments(self, patient_id: str) -> list[AssignmentPayload] | None:
        """Return every task dispatched to the patient, newest first."""

        record = self._patient_store.load(patient_id)
        if record is None:
            return None
        return [
            AssignmentPayload(
                assignment_id=assignment.assignment_id,
                task=cast(TaskName, assignment.task),
                task_params=assignment.task_params,
                due_date=assignment.due_date,
                status=cast(Literal["pending", "completed"], assignment.status),
                assigned_at=assignment.assigned_at,
                completed_at=assignment.completed_at,
                completed_session=assignment.completed_session,
            )
            for assignment in sorted(
                record.assignments, key=lambda item: item.assigned_at, reverse=True
            )
        ]

    def add_assignment(
        self,
        patient_id: str,
        task: str,
        task_params: Mapping[str, Any],
        due_date: str,
    ) -> AssignmentPayload | None:
        """Persist one therapist-dispatched task onto the patient record."""

        record = self._patient_store.load(patient_id)
        if record is None:
            return None
        assignment = TaskAssignment(
            assignment_id=uuid.uuid4().hex,
            task=task,
            task_params={str(key): value for key, value in task_params.items()},
            due_date=due_date,
            status="pending",
            assigned_at=time.time(),
        )
        record.assignments.append(assignment)
        try:
            self._patient_store.save(record)
        except OSError as error:
            LOGGER.warning("assignment_save_failed id=%s error=%s", patient_id, error)
        return AssignmentPayload(
            assignment_id=assignment.assignment_id,
            task=cast(TaskName, assignment.task),
            task_params=assignment.task_params,
            due_date=assignment.due_date,
            status=cast(Literal["pending", "completed"], assignment.status),
            assigned_at=assignment.assigned_at,
            completed_at=assignment.completed_at,
            completed_session=assignment.completed_session,
        )

    def _task_default_table(self) -> dict[str, dict[str, float]]:
        """Collect numeric defaults (speed/duration) per task for prescriptions."""

        table: dict[str, dict[str, float]] = {}
        for task_name, raw in self._task_config["tasks"].items():
            if not isinstance(raw, Mapping):
                continue
            table[str(task_name)] = {
                key: float(value)
                for key, value in raw.items()
                if key in ("reference_speed", "task_duration") and isinstance(value, (int, float))
            }
        return table

    def patient_assessment(self, patient_id: str) -> PatientAssessmentPayload | None:
        """Return the clinical agent's longitudinal trend assessment."""

        record = self._patient_store.load(patient_id)
        if record is None:
            return None
        assessment = assess_history(record.history)
        return PatientAssessmentPayload(
            patient_id=record.patient_id,
            sessions_analyzed=assessment.sessions_analyzed,
            classification=assessment.classification,
            score_slope=assessment.score_slope,
            completion_slope=assessment.completion_slope,
            error_slope=assessment.error_slope,
            avg_score_recent=assessment.avg_score_recent,
            avg_completion_recent=assessment.avg_completion_recent,
            avg_error_recent=assessment.avg_error_recent,
            flags=assessment.flags,
            narrative=assessment.narrative,
            risk_level=assessment.risk_level,
            evidence=assessment.evidence,
        )

    def patient_prescription(self, patient_id: str) -> SessionPrescriptionPayload | None:
        """Return the clinical agent's suggestion for the next session."""

        record = self._patient_store.load(patient_id)
        if record is None:
            return None
        prescription = recommend_next_session(
            record.history,
            record.profile,
            self._task_default_table(),
            vars(record.clinical_profile),
        )
        return SessionPrescriptionPayload(
            patient_id=record.patient_id,
            task=cast(TaskName, prescription.task),
            task_params=prescription.task_params,
            mode=cast(ControlMode, prescription.mode),
            difficulty_action=prescription.difficulty_action,
            rationale=prescription.rationale,
            risk_level=prescription.risk_level,
            confidence=prescription.confidence,
            missing_data=prescription.missing_data,
            precautions=prescription.precautions,
            requires_doctor_approval=prescription.requires_doctor_approval,
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
                self._queue_llm_task(
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

        loop = asyncio.get_running_loop()
        if self._llm_loop is not loop:
            if self._llm_worker_task is not None and not self._llm_worker_task.done():
                self._llm_worker_task.cancel()
            self._llm_loop = loop
            self._llm_queue = asyncio.Queue()
            self._llm_worker_task = None
        if self._llm_worker_task is None or self._llm_worker_task.done():
            self._llm_worker_task = asyncio.create_task(self._llm_worker_loop())

    def _queue_llm_task(self, task: _LLMTask) -> None:
        """Queue one optional LLM job on the current event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Unit tests and offline report generation may tick synchronously.
            # LLM enrichment is optional and must never make that path fail.
            return
        self._ensure_llm_worker()
        if self._llm_queue is not None:
            self._llm_queue.put_nowait(task)

    async def _llm_worker_loop(self) -> None:
        """Drain LLM jobs without ever touching the 20 Hz control loop."""

        while True:
            queue = self._llm_queue
            if queue is None:
                return
            task = await queue.get()
            try:
                await self._process_llm_task(task)
            except Exception:  # noqa: BLE001 - LLM failure must not propagate
                LOGGER.exception("llm_task_failed kind=%s", task.kind)
            finally:
                queue.task_done()

    async def aclose(self) -> None:
        """Cancel session-owned tasks and close optional network clients."""

        self._cancel_run_task()
        if self._llm_worker_task is not None and not self._llm_worker_task.done():
            self._llm_worker_task.cancel()
            try:
                await self._llm_worker_task
            except asyncio.CancelledError:
                pass
        self._llm_worker_task = None
        self._llm_queue = None
        self._llm_loop = None
        await self._llm_agent.client.aclose()

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
        record = self._patient_store.load(self._patient_id)
        if record is not None and record.history:
            trend = assess_history(record.history)
            context["patient_history"] = {
                "patient_id": record.patient_id,
                "session_count": record.session_count,
                "trend_classification": trend.classification,
                "recent_avg_completion_rate": (
                    round(trend.avg_completion_recent, 3)
                    if trend.avg_completion_recent is not None
                    else None
                ),
                "recent_avg_score": (
                    round(trend.avg_score_recent, 3) if trend.avg_score_recent is not None else None
                ),
                "recent_tasks": [entry.task for entry in record.history[-3:]],
                "assessment": trend.narrative,
            }
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
            patient_id=self._patient_id,
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
