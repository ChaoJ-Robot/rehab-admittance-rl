from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
for package in ("rehab_robot_bridge", "rehab_policy_node"):
    source = ROOT / "ros2_ws" / "src" / package
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from rehab_policy_node.controller import PolicyController  # noqa: E402
from rehab_robot_bridge.core import (  # noqa: E402
    AdmittanceCommand,
    RobotState,
    Ros2RobotAdapter,
    SimulationRobotAdapter,
    apply_low_speed_limit,
)
from rehab_robot_bridge.watchdog import CommunicationWatchdog, WatchdogDecision  # noqa: E402
from rehab_sim.config import load_yaml  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _state(sensor_ok: bool = True) -> RobotState:
    return RobotState(
        timestamp_s=0.0,
        joint_position=np.zeros(3),
        joint_velocity=np.zeros(3),
        end_effector_pose=np.array([0.35, 0.0, 0.0]),
        end_effector_velocity=np.array([0.02, 0.0, 0.0]),
        wrench=np.array([0.05, 0.0, 0.0]),
        sensor_ok=sensor_ok,
        source="test",
    )


def test_simulation_and_ros2_adapters_share_parameter_only_contract() -> None:
    command = AdmittanceCommand.from_vector([3.0, 3.0, 0.25, 0.0, 1.0], source="test_fixed")
    simulation = SimulationRobotAdapter()
    simulation.write_admittance_parameters(command)
    simulation.set_enabled(True)
    assert simulation.last_command == command
    assert simulation.enabled

    received: list[AdmittanceCommand] = []
    hardware = Ros2RobotAdapter(command_sink=received.append)
    assert not hardware.read_state().sensor_ok
    assert hardware.update_joint_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert hardware.update_end_effector([0.35, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert hardware.update_wrench([0.0, 0.0, 0.0])
    assert hardware.read_state().sensor_ok
    hardware.write_admittance_parameters(command)
    assert received == [command]
    assert not hasattr(hardware, "write_torque")


def test_watchdog_requires_fresh_pose_and_wrench() -> None:
    clock = FakeClock()
    watchdog = CommunicationWatchdog(
        {"pose": 0.1, "wrench": 0.1},
        clock=clock,
    )
    assert watchdog.evaluate().fallback_required
    watchdog.mark("pose")
    watchdog.mark("wrench")
    assert watchdog.evaluate().healthy
    clock.now = 0.11
    stale = watchdog.evaluate()
    assert not stale.healthy
    assert stale.fallback_required
    assert stale.reasons == ("pose_timeout", "wrench_timeout")


def test_low_speed_test_caps_only_task_space_velocity_scale() -> None:
    command = AdmittanceCommand.from_vector([3.0, 3.0, 0.25, 1.0, 1.0], source="test_policy")
    limited = apply_low_speed_limit(
        command,
        enabled=True,
        velocity_scale_limit=0.5,
    )
    np.testing.assert_allclose(limited.damping, command.damping)
    assert limited.assist_gain == command.assist_gain
    assert limited.velocity_scale == 0.5
    assert limited.low_speed_test


def test_ros2_config_has_explicit_hardware_gate_and_watchdog_rates() -> None:
    config = load_yaml(ROOT / "configs" / "ros2.yaml")["ros2"]
    assert config["hardware"]["enabled"] is False
    assert config["hardware"]["validation_required"] is True
    assert config["policy"]["deterministic"] is True
    assert float(config["rates"]["policy_update_hz"]) > 0.0
    assert float(config["watchdog"]["state_timeout_ms"]) > 0.0
    assert float(config["watchdog"]["wrench_timeout_ms"]) > 0.0
    assert config["low_speed_test"]["human_contact_allowed"] is False


def test_deterministic_policy_repeats_exactly_for_same_observation() -> None:
    from rehab_policy_node.controller import DeterministicAdmittancePolicy

    policy = DeterministicAdmittancePolicy()
    observation = {
        "pose_error": np.array([0.02, -0.01, 0.05]),
        "wrench": np.array([0.2, -0.1, 0.01]),
        "velocity": np.array([0.03, 0.0, 0.0]),
    }
    first = policy.predict(observation, deterministic=True)
    second = policy.predict(observation, deterministic=True)
    np.testing.assert_array_equal(first, second)


def test_policy_controller_keeps_fixed_and_adaptive_outputs_safe() -> None:
    controller = PolicyController(ROOT / "configs")
    healthy = WatchdogDecision(True, False, (), 0.01, {"pose": 0.01, "wrench": 0.01})

    fixed = controller.compute(
        _state(),
        [0.36, 0.01, 0.02],
        adaptive_enabled=False,
        watchdog=healthy,
    )
    assert fixed.command.source == "fixed_parameters"
    assert not fixed.command.fallback
    assert fixed.command.low_speed_test
    assert fixed.command.velocity_scale <= 0.5
    assert fixed.decision.action.shape == (4,)

    adaptive = controller.compute(
        _state(),
        [0.36, 0.01, 0.02],
        adaptive_enabled=True,
        watchdog=healthy,
    )
    assert adaptive.command.source == "deterministic_policy"
    assert adaptive.decision.raw_action.shape == (4,)
    assert adaptive.command.velocity_scale <= 0.5


def test_watchdog_failure_selects_fallback_without_policy_control() -> None:
    controller = PolicyController(ROOT / "configs")
    stale = WatchdogDecision(
        False,
        True,
        ("wrench_timeout",),
        0.2,
        {"pose": 0.01, "wrench": 0.2},
    )
    output = controller.compute(
        _state(),
        [0.35, 0.0, 0.0],
        adaptive_enabled=True,
        watchdog=stale,
    )
    assert output.command.source == "watchdog_fallback"
    assert output.command.fallback
    assert "communication_watchdog" in output.decision.reasons
