"""Common robot adapters and ROS 2 state-bridge primitives."""

from rehab_robot_bridge.core import (
    AdmittanceCommand,
    RobotAdapter,
    RobotState,
    Ros2RobotAdapter,
    SimulationRobotAdapter,
)
from rehab_robot_bridge.watchdog import CommunicationWatchdog, WatchdogDecision

__all__ = [
    "AdmittanceCommand",
    "CommunicationWatchdog",
    "RobotAdapter",
    "RobotState",
    "Ros2RobotAdapter",
    "SimulationRobotAdapter",
    "WatchdogDecision",
]
