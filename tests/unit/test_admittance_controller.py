from dataclasses import replace
from pathlib import Path

import numpy as np
from rehab_sim.config import load_yaml
from rehab_sim.controllers import AdmittanceController, AdmittanceParameters
from rehab_sim.robot.kinematics import WorkspaceBounds


def _controller() -> AdmittanceController:
    root = Path(__file__).parents[2]
    config = load_yaml(root / "configs" / "admittance.yaml")
    return AdmittanceController.from_config(
        config,
        workspace=WorkspaceBounds(x=(-1.0, 1.0), y=(-1.0, 1.0), theta=(-np.pi, np.pi)),
    )


def _update(controller: AdmittanceController, wrench: np.ndarray, count: int) -> list:
    output = []
    for _ in range(count):
        output.append(
            controller.update(
                measured_pose=controller.desired_pose,
                measured_velocity=controller.desired_velocity,
                reference_pose=np.zeros(3),
                reference_velocity=np.zeros(3),
                interaction_wrench=wrench,
            )
        )
    return output


def test_zero_force_has_no_drift() -> None:
    controller = _controller()
    controller.reset(np.zeros(3))
    outputs = _update(controller, np.zeros(3), 1000)

    np.testing.assert_allclose(outputs[-1].desired_pose, np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(outputs[-1].desired_velocity, np.zeros(3), atol=1.0e-12)


def test_force_low_pass_and_soft_deadzone() -> None:
    controller = _controller()
    controller.reset(np.zeros(3))
    output = _update(controller, np.array([0.05, 0.0, 0.0]), 100)[-1]

    assert 0.0 < output.filtered_wrench[0] < 0.05
    assert output.effective_wrench[0] == 0.0


def test_force_direction_and_reverse_response() -> None:
    positive = _controller()
    positive.reset(np.zeros(3))
    positive_output = _update(positive, np.array([0.8, 0.0, 0.0]), 500)[-1]

    negative = _controller()
    negative.reset(np.zeros(3))
    negative_output = _update(negative, np.array([-0.8, 0.0, 0.0]), 500)[-1]

    assert positive_output.desired_pose[0] > 0.0
    assert negative_output.desired_pose[0] < 0.0
    assert positive_output.desired_velocity[0] > 0.0
    assert negative_output.desired_velocity[0] < 0.0


def test_increasing_damping_reduces_transient_velocity() -> None:
    base = _controller()
    high_damping = AdmittanceController(
        replace(base.parameters, damping=base.parameters.damping * 2.0),
        workspace=base.workspace,
    )
    base.reset(np.zeros(3))
    high_damping.reset(np.zeros(3))

    base_output = _update(base, np.array([0.8, 0.0, 0.0]), 200)[-1]
    high_output = _update(high_damping, np.array([0.8, 0.0, 0.0]), 200)[-1]

    assert high_output.desired_velocity[0] < base_output.desired_velocity[0]


def test_velocity_and_acceleration_limits_are_respected() -> None:
    controller = _controller()
    controller.reset(np.zeros(3))
    previous_velocity = np.zeros(3)
    for output in _update(controller, np.array([100.0, 100.0, 10.0]), 1000):
        limits = controller.parameters.velocity_limits * controller.parameters.velocity_scale
        assert np.all(np.abs(output.desired_velocity) <= limits + 1.0e-12)
        acceleration = (
            output.desired_velocity - previous_velocity
        ) / controller.parameters.sample_time_s
        assert np.all(np.abs(acceleration) <= controller.parameters.acceleration_limits + 1.0e-9)
        previous_velocity = output.desired_velocity


def test_invalid_parameters_are_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        AdmittanceParameters(
            sample_time_s=0.0,
            mass=np.ones(3),
            damping=np.ones(3),
            stiffness=np.ones(3),
            assist_gain=0.0,
            assist_damping=0.0,
            velocity_limits=np.ones(3),
            acceleration_limits=np.ones(3),
            force_filter_time_constant_s=np.ones(3),
            force_deadzone=np.zeros(3),
        )
