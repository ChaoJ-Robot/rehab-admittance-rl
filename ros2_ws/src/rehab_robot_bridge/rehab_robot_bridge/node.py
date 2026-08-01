"""ROS 2 state bridge for real-driver and simulation-compatible messages."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rehab_interfaces.msg import (
    AdmittanceParameters,
    EndEffectorState,
    RealtimeMetrics,
    WrenchState,
)
from sensor_msgs.msg import JointState

from rehab_robot_bridge.config import load_ros_config
from rehab_robot_bridge.core import AdmittanceCommand, Ros2RobotAdapter

LOGGER = logging.getLogger("rehab.ros2.robot_bridge")


def _topic(config: dict[str, Any], name: str, default: str) -> str:
    topics = config.get("topics", {})
    value = topics.get(name, default) if isinstance(topics, dict) else default
    return str(value)


class RobotBridgeNode(Node):
    """Normalize hardware-driver ROS messages into the project state contract.

    The bridge subscribes to joint/pose/wrench data and admittance parameters.
    Parameter messages are retained for a downstream validated driver; this
    node deliberately has no torque, current, or motor-command publisher.
    """

    def __init__(self) -> None:
        super().__init__("rehab_robot_bridge")
        self.declare_parameter("config_dir", "")
        config_dir = str(self.get_parameter("config_dir").value)
        try:
            self._config = load_ros_config(config_dir)
        except Exception as error:  # noqa: BLE001 - startup must fail clearly
            self.get_logger().fatal(f"cannot load ROS 2 config: {error}")
            raise
        self._adapter = Ros2RobotAdapter(command_sink=self._on_parameter_command)
        self._metrics_pub = self.create_publisher(
            RealtimeMetrics,
            _topic(self._config, "realtime_metrics", "/rehab/metrics/realtime"),
            10,
        )
        self._joint_sub = self.create_subscription(
            JointState,
            _topic(self._config, "joint_states", "/joint_states"),
            self._on_joint_state,
            10,
        )
        self._ee_sub = self.create_subscription(
            EndEffectorState,
            _topic(self._config, "end_effector_state", "/rehab_robot/end_effector_state"),
            self._on_end_effector_state,
            10,
        )
        self._wrench_sub = self.create_subscription(
            WrenchState,
            _topic(self._config, "wrench", "/rehab_robot/wrench"),
            self._on_wrench,
            10,
        )
        self._parameter_sub = self.create_subscription(
            AdmittanceParameters,
            _topic(self._config, "admittance_parameters", "/rehab/admittance/parameters"),
            self._on_parameters,
            10,
        )
        rates = self._config.get("rates", {})
        publish_hz = float(rates.get("state_publish_hz", 20.0)) if isinstance(rates, dict) else 20.0
        if publish_hz <= 0.0:
            raise ValueError("state_publish_hz must be positive")
        self._timer = self.create_timer(1.0 / publish_hz, self._publish_metrics)
        self.get_logger().info("robot bridge ready; parameter output is task-space only")

    def _on_joint_state(self, message: JointState) -> None:
        positions = np.asarray(list(message.position[:3]), dtype=np.float64)
        velocities = np.asarray(list(message.velocity[:3]), dtype=np.float64)
        if positions.shape != (3,):
            self.get_logger().warning("joint state requires three planar joints")
            return
        if velocities.shape != (3,):
            velocities = np.zeros(3, dtype=np.float64)
        self._adapter.update_joint_state(positions, velocities)

    def _on_end_effector_state(self, message: EndEffectorState) -> None:
        self._adapter.update_end_effector(message.pose, message.velocity, message.valid)

    def _on_wrench(self, message: WrenchState) -> None:
        self._adapter.update_wrench(message.wrench, message.valid)

    def _on_parameters(self, message: AdmittanceParameters) -> None:
        try:
            command = AdmittanceCommand(
                damping=message.damping,
                assist_gain=float(message.assist_gain),
                velocity_scale=float(message.velocity_scale),
                source=message.source or "ros2_policy",
                fallback=bool(message.fallback),
                low_speed_test=bool(message.low_speed_test),
            )
            self._adapter.write_admittance_parameters(command)
        except (TypeError, ValueError) as error:
            self.get_logger().error(f"rejected invalid admittance parameters: {error}")

    def _on_parameter_command(self, command: AdmittanceCommand) -> None:
        LOGGER.info(
            "admittance_parameter_command source=%s fallback=%s low_speed_test=%s values=%s",
            command.source,
            command.fallback,
            command.low_speed_test,
            command.as_vector().tolist(),
        )

    def _publish_metrics(self) -> None:
        state = self._adapter.read_state()
        message = RealtimeMetrics()
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose = state.end_effector_pose.tolist()
        message.pose_error = [0.0, 0.0, 0.0]
        message.velocity = state.end_effector_velocity.tolist()
        message.interaction_wrench = state.wrench.tolist()
        message.human_power_w = max(0.0, float(np.dot(state.wrench, state.end_effector_velocity)))
        message.fatigue = 0.0
        message.task_progress = 0.0
        message.sensor_ok = state.sensor_ok
        message.safety_status = "sensor_ok" if state.sensor_ok else "sensor_unavailable"
        message.source = state.source
        self._metrics_pub.publish(message)


def main(args: list[str] | None = None) -> None:
    """Run the ROS 2 robot bridge node."""

    rclpy.init(args=args)
    node = RobotBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
