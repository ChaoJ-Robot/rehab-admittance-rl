"""Task lifecycle node with no robot-level control authority."""

from __future__ import annotations

from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rehab_interfaces.msg import RehabTaskState
from rehab_interfaces.srv import (
    EnableRl,
    LoadPolicy,
    PauseTask,
    ResetTask,
    SetPatientProfile,
    StartTask,
    StopTask,
)
from rehab_robot_bridge.config import find_config_dir, load_ros_config


def _topic(config: dict[str, Any], name: str, default: str) -> str:
    topics = config.get("topics", {})
    return str(topics.get(name, default)) if isinstance(topics, dict) else default


class TaskManagerNode(Node):
    """Expose task services and publish state consumed by policy/UI nodes."""

    def __init__(self) -> None:
        super().__init__("rehab_task_manager")
        self.declare_parameter("config_dir", "")
        config_dir = find_config_dir(str(self.get_parameter("config_dir").value))
        self._config = load_ros_config(str(config_dir))
        self._task_id = "none"
        self._task_name = "point_to_point"
        self._patient_profile = "moderate"
        self._state = "idle"
        self._elapsed_s = 0.0
        self._duration_s = 0.0
        self._progress = 0.0
        self._rl_enabled = False
        self._state_pub = self.create_publisher(
            RehabTaskState,
            _topic(self._config, "system_state", "/rehab/system/state"),
            10,
        )
        self._status_pub = self.create_publisher(
            RehabTaskState,
            _topic(self._config, "task_status", "/rehab/task/status"),
            10,
        )
        self.create_service(StartTask, "/rehab/start_task", self._start)
        self.create_service(PauseTask, "/rehab/pause_task", self._pause)
        self.create_service(StopTask, "/rehab/stop_task", self._stop)
        self.create_service(ResetTask, "/rehab/reset_task", self._reset)
        self.create_service(EnableRl, "/rehab/enable_rl", self._enable_rl)
        self.create_service(LoadPolicy, "/rehab/load_policy", self._load_policy)
        self.create_service(SetPatientProfile, "/rehab/set_patient_profile", self._set_profile)
        rates = self._config.get("rates", {})
        publish_hz = float(rates.get("state_publish_hz", 20.0)) if isinstance(rates, dict) else 20.0
        if publish_hz <= 0.0:
            raise ValueError("state_publish_hz must be positive")
        self._timer = self.create_timer(1.0 / publish_hz, self._tick)

    def _start(
        self, request: StartTask.Request, response: StartTask.Response
    ) -> StartTask.Response:
        if self._state in ("running", "paused"):
            response.accepted = False
            response.message = "a task is already active"
            return response
        if request.duration_s <= 0.0:
            response.accepted = False
            response.message = "duration_s must be positive"
            return response
        self._task_id = request.task_id or "ros2_task"
        self._task_name = self._task_id
        self._patient_profile = request.patient_profile or "moderate"
        self._duration_s = float(request.duration_s)
        self._elapsed_s = 0.0
        self._progress = 0.0
        self._state = "running"
        response.accepted = True
        response.message = "task started"
        return response

    def _pause(
        self, _request: PauseTask.Request, response: PauseTask.Response
    ) -> PauseTask.Response:
        if self._state != "running":
            response.accepted = False
            response.message = "only a running task can be paused"
            return response
        self._state = "paused"
        response.accepted = True
        response.message = "task paused"
        return response

    def _stop(self, _request: StopTask.Request, response: StopTask.Response) -> StopTask.Response:
        if self._state not in ("running", "paused"):
            response.accepted = False
            response.message = "no active task"
            return response
        self._state = "stopped"
        response.accepted = True
        response.message = "task stopped"
        return response

    def _reset(
        self, _request: ResetTask.Request, response: ResetTask.Response
    ) -> ResetTask.Response:
        self._task_id = "none"
        self._task_name = "point_to_point"
        self._elapsed_s = 0.0
        self._duration_s = 0.0
        self._progress = 0.0
        self._state = "idle"
        response.accepted = True
        response.message = "task reset"
        return response

    def _enable_rl(
        self, request: EnableRl.Request, response: EnableRl.Response
    ) -> EnableRl.Response:
        policy = self._config.get("policy", {})
        allow = bool(policy.get("allow_rl", False)) if isinstance(policy, dict) else False
        if request.enabled and not allow:
            response.accepted = False
            response.message = "adaptive policy is disabled by ros2.yaml"
            return response
        self._rl_enabled = bool(request.enabled)
        response.accepted = True
        response.message = "adaptive policy state updated"
        return response

    def _load_policy(
        self, request: LoadPolicy.Request, response: LoadPolicy.Response
    ) -> LoadPolicy.Response:
        response.accepted = False
        response.message = (
            "online model loading is disabled; use the deterministic policy "
            "and validated deployment"
            if request.path
            else "policy path is empty"
        )
        return response

    def _set_profile(
        self, request: SetPatientProfile.Request, response: SetPatientProfile.Response
    ) -> SetPatientProfile.Response:
        if self._state in ("running", "paused"):
            response.accepted = False
            response.message = "patient profile cannot change during an active task"
            return response
        if not request.profile:
            response.accepted = False
            response.message = "profile must not be empty"
            return response
        self._patient_profile = request.profile
        response.accepted = True
        response.message = "patient profile updated"
        return response

    def _tick(self) -> None:
        if self._state == "running":
            self._elapsed_s = min(self._duration_s, self._elapsed_s + 1.0 / 20.0)
            self._progress = 1.0 if self._duration_s <= 0.0 else self._elapsed_s / self._duration_s
            if self._elapsed_s >= self._duration_s:
                self._state = "completed"
        message = RehabTaskState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.task_id = self._task_id
        message.task_name = self._task_name
        message.patient_profile = self._patient_profile
        message.state = self._state
        message.progress = self._progress
        message.elapsed_s = self._elapsed_s
        message.enabled = self._state in ("running", "paused")
        message.rl_enabled = self._rl_enabled
        self._state_pub.publish(message)
        self._status_pub.publish(message)


def main(args: list[str] | None = None) -> None:
    """Run the task manager node."""

    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
