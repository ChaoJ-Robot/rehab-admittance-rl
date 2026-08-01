from pathlib import Path

import numpy as np
from rehab_sim.config import load_yaml
from rehab_sim.robot.kinematics import Planar3RGeometry, WorkspaceBounds


def _geometry() -> Planar3RGeometry:
    root = Path(__file__).parents[2]
    return Planar3RGeometry.from_config(load_yaml(root / "configs" / "robot.yaml"))


def test_zero_pose_has_collinear_joint_positions() -> None:
    geometry = _geometry()
    joint1, joint2, joint3 = geometry.joint_positions([0.0, 0.0, 0.0])
    first = joint2 - joint1
    second = joint3 - joint2
    cross = first[0] * second[1] - first[1] * second[0]

    assert abs(cross) < 1.0e-10


def test_jacobian_matches_forward_difference() -> None:
    geometry = _geometry()
    qpos = np.array([0.2, -0.3, 0.4])
    epsilon = 1.0e-7
    numerical = np.zeros((3, 3))
    for index in range(3):
        plus = qpos.copy()
        minus = qpos.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[:, index] = (geometry.forward(plus) - geometry.forward(minus)) / (2.0 * epsilon)
    numerical[2, :] = 1.0

    np.testing.assert_allclose(geometry.jacobian(qpos), numerical, atol=1.0e-7)


def test_workspace_bounds_are_enforced() -> None:
    bounds = WorkspaceBounds(x=(-1.0, 1.0), y=(-1.0, 1.0), theta=(-3.14, 3.14))

    assert bounds.contains([0.0, 0.0, 0.0])
    assert not bounds.contains([1.01, 0.0, 0.0])
