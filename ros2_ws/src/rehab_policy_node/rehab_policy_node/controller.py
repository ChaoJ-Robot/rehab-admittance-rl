"""Deterministic policy and safety composition for the ROS 2 node."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rehab_robot_bridge.config import load_ros_config
from rehab_robot_bridge.core import (
    AdmittanceCommand,
    RobotState,
    apply_low_speed_limit,
)
from rehab_robot_bridge.watchdog import WatchdogDecision

FloatArray = NDArray[np.float64]


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


class DeterministicAdmittancePolicy:
    """A deterministic parameter-only policy for low-speed deployment.

    The output is a normalized four-element parameter increment
    ``[dDxy,dDtheta,dKa,dlambda_v]``. It is never a joint command or motor
    torque. The safety supervisor remains the only authority that approves it.
    """

    def predict(self, observation: Any, deterministic: bool = True) -> FloatArray:
        del deterministic
        error = _vector(observation["pose_error"], 3, "pose_error")
        wrench = _vector(observation["wrench"], 3, "wrench")
        velocity = _vector(observation["velocity"], 3, "velocity")
        force_norm = float(np.linalg.norm(wrench[:2]))
        error_norm = float(np.linalg.norm(error[:2]))
        speed_norm = float(np.linalg.norm(velocity))
        # Conservative, deterministic increments. They adapt only D/Ka/lambda_v.
        return np.clip(
            np.asarray(
                [
                    0.15 * (force_norm - 0.15),
                    0.05 * abs(error[2]),
                    0.10 * error_norm,
                    -0.10 * speed_norm,
                ],
                dtype=np.float64,
            ),
            -1.0,
            1.0,
        )


@dataclass(frozen=True)
class PolicyOutput:
    """Auditable output of one policy/safety cycle."""

    command: AdmittanceCommand
    decision: Any
    watchdog: WatchdogDecision


class PolicyController:
    """Compose fixed mode, deterministic policy, safety and low-speed cap."""

    def __init__(self, config_dir: str | Path) -> None:
        directory = Path(config_dir)
        # The safety layer is the existing project package. Add the repository
        # root when a ROS 2 executable is launched from a colcon install,
        # where the current working directory is not on sys.path.
        import sys

        project_root = directory.resolve().parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from rehab_sim.config import load_yaml
        from rehab_sim.controllers import AdmittanceParameters
        from rehab_sim.safety import (
            SafetySupervisor,
            load_safety_configuration,
        )
        from rehab_sim.safety.parameter_projector import parameter_vector
        from rehab_sim.safety.policy_runtime import SafePolicyRuntime

        admittance_config = load_yaml(directory / "admittance.yaml")
        safety_config = load_yaml(directory / "safety.yaml")
        self.ros_config = load_ros_config(str(directory))
        self.supervisor = SafetySupervisor(
            load_safety_configuration(admittance_config, safety_config)
        )
        low_speed = self.ros_config.get("low_speed_test", {})
        if isinstance(low_speed, dict) and bool(low_speed.get("enabled", False)):
            velocity_limit = float(low_speed.get("velocity_scale_limit", 1.0))
            safe_lower = float(self.supervisor.configuration.parameter_lower[4])
            if velocity_limit < safe_lower:
                raise ValueError(
                    "low_speed_test.velocity_scale_limit must remain inside safety parameter bounds"
                )
        self._baseline = parameter_vector(AdmittanceParameters.from_config(admittance_config))
        self._current = self._baseline.copy()
        self._runtime = SafePolicyRuntime(
            self.supervisor,
            policy=DeterministicAdmittancePolicy(),
        )

    @property
    def current_parameters(self) -> FloatArray:
        return self._current.copy()

    @property
    def baseline_parameters(self) -> FloatArray:
        return self._baseline.copy()

    def reset(self) -> None:
        self._current = self._baseline.copy()

    @staticmethod
    def _safety_observation(state: RobotState) -> Any:
        from rehab_sim.safety import SafetyObservation

        return SafetyObservation(
            interaction_wrench=state.wrench,
            task_velocity=state.end_effector_velocity,
            task_acceleration=None,
            sensor_ok=state.sensor_ok,
        )

    def compute(
        self,
        state: RobotState,
        reference_pose: ArrayLike,
        *,
        adaptive_enabled: bool,
        watchdog: WatchdogDecision,
    ) -> PolicyOutput:
        """Produce one parameter-only command through the independent safety layer."""

        reference = _vector(reference_pose, 3, "reference_pose")
        safety_observation = self._safety_observation(state)
        pose_error = reference - state.end_effector_pose
        pose_error[2] = float(np.arctan2(np.sin(pose_error[2]), np.cos(pose_error[2])))
        observation = {
            "pose_error": pose_error,
            "wrench": state.wrench,
            "velocity": state.end_effector_velocity,
        }
        if watchdog.fallback_required:
            decision = self.supervisor.fallback(
                self._current,
                np.zeros(4, dtype=np.float64),
                tuple(["communication_watchdog", *watchdog.reasons]),
                safety_observation,
            )
            source = "watchdog_fallback"
        elif adaptive_enabled:
            decision = self._runtime.act(observation, self._current, safety_observation)
            source = "deterministic_policy" if not decision.fallback else "policy_fallback"
        else:
            decision = self.supervisor.supervise(
                np.zeros(4, dtype=np.float64),
                self._current,
                safety_observation,
                inference_elapsed_s=0.0,
                model_loaded=True,
            )
            source = "fixed_parameters" if not decision.fallback else "fixed_mode_fallback"
        command = AdmittanceCommand.from_vector(
            decision.parameters,
            source=source,
            fallback=decision.fallback,
        )
        low_speed = self.ros_config.get("low_speed_test", {})
        if not isinstance(low_speed, dict):
            low_speed = {}
        low_speed_enabled = bool(low_speed.get("enabled", False))
        velocity_limit = float(low_speed.get("velocity_scale_limit", 1.0))
        command = apply_low_speed_limit(
            command,
            enabled=low_speed_enabled,
            velocity_scale_limit=velocity_limit,
        )
        self._current = command.as_vector()
        return PolicyOutput(command=command, decision=decision, watchdog=watchdog)
