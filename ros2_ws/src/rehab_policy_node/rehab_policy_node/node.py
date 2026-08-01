"""ROS 2 deterministic policy node with an independent communication watchdog."""

from __future__ import annotations

from typing import Any

import rclpy
from geometry_msgs.msg import Pose2D
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rehab_interfaces.msg import (
    AdmittanceParameters,
    EndEffectorState,
    PolicyAction,
    RehabTaskState,
    SafetyState,
    WrenchState,
)
from rehab_robot_bridge.config import find_config_dir, load_ros_config
from rehab_robot_bridge.core import Ros2RobotAdapter
from rehab_robot_bridge.watchdog import CommunicationWatchdog
from sensor_msgs.msg import JointState

from rehab_policy_node.controller import PolicyController


def _topic(config: dict[str, Any], name: str, default: str) -> str:
    topics = config.get("topics", {})
    return str(topics.get(name, default)) if isinstance(topics, dict) else default


class PolicyNode(Node):
    """Publish only supervised admittance parameters and audit messages."""

    def __init__(self) -> None:
        super().__init__("rehab_policy_node")
        self.declare_parameter("config_dir", "")
        config_dir = find_config_dir(str(self.get_parameter("config_dir").value))
        self._config = load_ros_config(str(config_dir))
        self._controller = PolicyController(config_dir)
        self._adapter = Ros2RobotAdapter()
        watchdog_config = self._config.get("watchdog", {})
        if not isinstance(watchdog_config, dict):
            raise ValueError("ros2.watchdog must be a mapping")
        self._watchdog = CommunicationWatchdog(
            {
                "pose": float(watchdog_config.get("state_timeout_ms", 100.0)) / 1000.0,
                "wrench": float(watchdog_config.get("wrench_timeout_ms", 100.0)) / 1000.0,
            },
            fallback_on_timeout=bool(watchdog_config.get("fallback_on_timeout", True)),
        )
        self._reference = [0.0, 0.0, 0.0]
        policy_config = self._config.get("policy", {})
        if isinstance(policy_config, dict):
            self._allow_adaptive = bool(policy_config.get("allow_rl", False))
        else:
            self._allow_adaptive = False
        self._adaptive_enabled = (
            str(self._config.get("mode", "fixed")) == "rl" and self._allow_adaptive
        )
        self._parameter_pub = self.create_publisher(
            AdmittanceParameters,
            _topic(self._config, "admittance_parameters", "/rehab/admittance/parameters"),
            10,
        )
        self._action_pub = self.create_publisher(
            PolicyAction,
            _topic(self._config, "policy_action", "/rehab/policy/action"),
            10,
        )
        self._safety_pub = self.create_publisher(
            SafetyState,
            _topic(self._config, "safety_status", "/rehab/safety/status"),
            10,
        )
        self.create_subscription(
            JointState,
            _topic(self._config, "joint_states", "/joint_states"),
            self._on_joint_state,
            10,
        )
        self.create_subscription(
            EndEffectorState,
            _topic(self._config, "end_effector_state", "/rehab_robot/end_effector_state"),
            self._on_end_effector,
            10,
        )
        self.create_subscription(
            WrenchState,
            _topic(self._config, "wrench", "/rehab_robot/wrench"),
            self._on_wrench,
            10,
        )
        self.create_subscription(
            Pose2D,
            _topic(self._config, "task_reference", "/rehab/task/reference"),
            self._on_reference,
            10,
        )
        self.create_subscription(
            RehabTaskState,
            _topic(self._config, "system_state", "/rehab/system/state"),
            self._on_task_state,
            10,
        )
        rates = self._config.get("rates", {})
        policy_hz = float(rates.get("policy_update_hz", 5.0)) if isinstance(rates, dict) else 5.0
        if policy_hz <= 0.0:
            raise ValueError("policy_update_hz must be positive")
        self._timer = self.create_timer(1.0 / policy_hz, self._control_tick)
        mode = "adaptive" if self._adaptive_enabled else "fixed"
        self.get_logger().info(
            f"policy node ready in {mode} mode; outputs are admittance parameters only"
        )

    def _on_joint_state(self, message: JointState) -> None:
        positions = list(message.position[:3])
        velocities = list(message.velocity[:3])
        if len(positions) != 3:
            self.get_logger().warning("joint state requires three planar joints")
            return
        if len(velocities) != 3:
            velocities = [0.0, 0.0, 0.0]
        self._adapter.update_joint_state(positions, velocities)

    def _on_end_effector(self, message: EndEffectorState) -> None:
        self._adapter.update_end_effector(message.pose, message.velocity, message.valid)
        if message.valid:
            self._watchdog.mark("pose")

    def _on_wrench(self, message: WrenchState) -> None:
        self._adapter.update_wrench(message.wrench, message.valid)
        if message.valid:
            self._watchdog.mark("wrench")

    def _on_reference(self, message: Pose2D) -> None:
        self._reference = [float(message.x), float(message.y), float(message.theta)]

    def _on_task_state(self, message: RehabTaskState) -> None:
        if self._allow_adaptive:
            self._adaptive_enabled = bool(message.rl_enabled)

    def _control_tick(self) -> None:
        state = self._adapter.read_state()
        watchdog = self._watchdog.evaluate()
        output = self._controller.compute(
            state,
            self._reference,
            adaptive_enabled=self._adaptive_enabled,
            watchdog=watchdog,
        )
        stamp = self.get_clock().now().to_msg()
        parameter_message = AdmittanceParameters()
        parameter_message.header.stamp = stamp
        parameter_message.damping = output.command.damping.tolist()
        parameter_message.assist_gain = output.command.assist_gain
        parameter_message.velocity_scale = output.command.velocity_scale
        parameter_message.fallback = output.command.fallback
        parameter_message.low_speed_test = output.command.low_speed_test
        parameter_message.source = output.command.source
        self._parameter_pub.publish(parameter_message)

        action_message = PolicyAction()
        action_message.header.stamp = stamp
        action_message.raw_action = output.decision.raw_action.tolist()
        action_message.safe_action = output.decision.action.tolist()
        action_message.approved = output.decision.approved
        action_message.fallback = output.decision.fallback
        action_message.inference_latency_s = 0.0
        action_message.reasons = list(output.decision.reasons)
        self._action_pub.publish(action_message)

        safety_message = SafetyState()
        safety_message.header.stamp = stamp
        safety_message.safe = (
            output.watchdog.healthy and state.sensor_ok and not output.decision.fallback
        )
        safety_message.fallback_active = output.command.fallback
        safety_message.sensor_ok = state.sensor_ok
        safety_message.state_age_s = output.watchdog.max_age_s
        safety_message.reasons = list(output.watchdog.reasons) + list(output.decision.reasons)
        self._safety_pub.publish(safety_message)


def main(args: list[str] | None = None) -> None:
    """Run the policy node."""

    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
