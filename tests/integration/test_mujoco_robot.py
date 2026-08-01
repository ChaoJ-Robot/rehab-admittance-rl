from pathlib import Path

import numpy as np
import pytest
from rehab_sim.config import load_yaml
from rehab_sim.robot import MujocoPlanarRobot, Planar3RGeometry, WorkspaceBounds


def _robot() -> MujocoPlanarRobot:
    root = Path(__file__).parents[2]
    config = load_yaml(root / "configs" / "robot.yaml")
    geometry = Planar3RGeometry.from_config(config)
    model_path = root / str(config["robot"]["mujoco_model"])
    return MujocoPlanarRobot(
        model_path,
        geometry,
        WorkspaceBounds(x=(-1.0, 1.0), y=(-1.0, 1.0), theta=(-np.pi, np.pi)),
    )


def test_mujoco_has_three_independent_controlled_joints() -> None:
    robot = _robot()
    target = np.array([0.25, -0.15, 0.2])
    np.testing.assert_allclose(robot.set_joint_targets(target), target)
    assert [robot.model.actuator_trnid[index, 0] for index in robot.actuator_ids] == list(
        robot.joint_ids
    )
    robot.step(1000)

    assert robot.model.nq == 3
    assert robot.model.nv == 3
    assert robot.model.nu == 3
    np.testing.assert_allclose(robot.qpos, target, atol=2.0e-2)
    assert np.all(np.isfinite(robot.data.qpos))
    assert np.all(np.isfinite(robot.data.qvel))


def test_site_pose_matches_analytic_kinematics() -> None:
    robot = _robot()
    qpos = np.array([0.2, -0.3, 0.4])
    robot.set_joint_state(qpos)

    np.testing.assert_allclose(robot.end_effector_pose, robot.geometry.forward(qpos), atol=1.0e-6)


def test_positive_x_force_has_expected_generalized_force_direction() -> None:
    robot = _robot()
    robot.set_joint_state([0.0, 0.0, 0.0])
    wrench = np.array([1.0, 0.0, 0.0])
    robot.set_external_wrench(wrench)
    expected = robot.geometry.jacobian([0.0, 0.0, 0.0]).T @ wrench

    np.testing.assert_allclose(robot.data.qfrc_applied, expected, atol=1.0e-6)


def test_joint_target_and_workspace_limits_are_effective() -> None:
    robot = _robot()
    clipped = robot.set_joint_targets([4.0, -4.0, 4.0])
    assert np.all(clipped <= np.pi)
    assert np.all(clipped >= -np.pi)
    robot.step(1000)

    assert robot.within_joint_limits()
    assert robot.within_workspace([0.0, 0.0, 0.0])
    assert not robot.within_workspace([1.01, 0.0, 0.0])

    robot.workspace = WorkspaceBounds(x=(-0.04, -0.02), y=(-0.7, -0.5), theta=(-0.1, 0.1))
    robot.set_joint_targets([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="outside the configured workspace"):
        robot.set_joint_targets([0.0, 0.0, 1.0])
