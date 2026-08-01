"""Gymnasium base environment for planar rehabilitation training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from rehab_sim.config import load_yaml
from rehab_sim.controllers import AdmittanceController, AdmittanceParameters
from rehab_sim.patients import ImpedancePatient, PatientProfile, load_patient_profiles
from rehab_sim.rewards import RewardComponents, RewardWeights
from rehab_sim.robot import (
    DampedLeastSquaresIK,
    MujocoPlanarRobot,
    Planar3RGeometry,
    WorkspaceBounds,
)
from rehab_sim.robot.kinematics import wrap_angle
from rehab_sim.tasks import (
    CircleTrajectory,
    FigureEightTrajectory,
    PointToPointTrajectory,
    ReferenceTrajectory,
)

FloatArray = NDArray[np.float64]


class PlanarRehabEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """A finite-horizon task environment around the Phase 1/2 simulation.

    Observation task vectors use ``[x,y,theta]`` in m, m, rad and interaction
    force vectors use ``[Fx,Fy,Tz]`` in N, N, N*m. Each action is a normalized
    low-frequency increment ``[dDxy,dDtheta,dKa,dlambda_v]``; it never writes
    an actuator torque or current.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        task_name: str,
        patient_profile: str = "moderate",
        seed: int = 0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode is not None:
            raise ValueError("Phase 4 environment currently supports render_mode=None only")
        self.task_name = task_name
        self.patient_profile_name = patient_profile
        self._seed_value = seed
        self.render_mode = render_mode
        self._root = Path(__file__).resolve().parents[2]
        self._robot_config = load_yaml(self._root / "configs" / "robot.yaml")
        self._admittance_config = load_yaml(self._root / "configs" / "admittance.yaml")
        self._patient_config = load_yaml(self._root / "configs" / "patient_profiles.yaml")
        self._task_config = load_yaml(self._root / "configs" / "tasks.yaml")
        self._rl_config = load_yaml(self._root / "configs" / "rl_sac.yaml")
        self._safety_config = load_yaml(self._root / "configs" / "safety.yaml")
        self._profiles = load_patient_profiles(self._patient_config)
        if patient_profile not in self._profiles:
            raise ValueError(f"unknown patient profile: {patient_profile}")
        self._profile: PatientProfile = self._profiles[patient_profile]
        self._geometry = Planar3RGeometry.from_config(self._robot_config)
        self._workspace = WorkspaceBounds.from_config(self._robot_config)
        model_path = self._root / str(self._robot_config["robot"]["mujoco_model"])
        self.robot = MujocoPlanarRobot(model_path, self._geometry, workspace=self._workspace)
        self._base_parameters = AdmittanceParameters.from_config(self._admittance_config)
        self.controller = AdmittanceController(self._base_parameters, workspace=self._workspace)
        self._patient = ImpedancePatient(
            self._profile,
            sample_time_s=float(self._base_parameters.sample_time_s),
            seed=seed,
        )
        self._ik = DampedLeastSquaresIK(
            self._geometry,
            self.robot.joint_ranges[:, 0],
            self.robot.joint_ranges[:, 1],
        )
        self._reward_weights = RewardWeights.from_config(self._rl_config)
        self._task: ReferenceTrajectory | None = None
        self._elapsed_s = 0.0
        self._last_progress = 0.0
        self._last_velocity = np.zeros(3, dtype=np.float64)
        self._last_force = np.zeros(3, dtype=np.float64)
        self._stagnation_s = 0.0
        self._episode_done = False
        self._last_reward_components = RewardComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._physics_dt = float(self.robot.model.opt.timestep)
        update_section = self._admittance_config["adaptive_update"]
        self._action_interval_s = 1.0 / float(
            self._rl_config.get("training", {}).get("parameter_update_hz", 5.0)
        )
        self._physics_steps_per_action = max(1, round(self._action_interval_s / self._physics_dt))
        self._action_scales = np.array(
            [
                float(update_section["action_scales"]["damping_xy"]),
                float(update_section["action_scales"]["damping_theta"]),
                float(update_section["action_scales"]["assist_gain"]),
                float(update_section["action_scales"]["velocity_scale"]),
            ],
            dtype=np.float64,
        )
        bounds = update_section["parameter_bounds"]
        self._parameter_lower = np.array(
            [
                bounds["damping_x"][0],
                bounds["damping_y"][0],
                bounds["damping_theta"][0],
                bounds["assist_gain"][0],
                bounds["velocity_scale"][0],
            ],
            dtype=np.float64,
        )
        self._parameter_upper = np.array(
            [
                bounds["damping_x"][1],
                bounds["damping_y"][1],
                bounds["damping_theta"][1],
                bounds["assist_gain"][1],
                bounds["velocity_scale"][1],
            ],
            dtype=np.float64,
        )
        self._simulation_limits = self._safety_config["simulation_limits"]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "joint_position": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "joint_velocity": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "end_effector_pose": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "end_effector_velocity": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "interaction_force": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "reference_pose": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "tracking_error": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "admittance_parameters": spaces.Box(0.0, np.inf, shape=(5,), dtype=np.float32),
                "task_progress": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "safety_status": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

    def _build_task(self, start_pose: FloatArray) -> ReferenceTrajectory:
        raw = self._task_config["tasks"][self.task_name]
        if self.task_name == "point_to_point":
            return PointToPointTrajectory(
                start_pose,
                float(raw["target_distance"]),
                float(raw["task_duration"]),
                float(raw["target_radius"]),
            )
        if self.task_name == "circle_tracking":
            return CircleTrajectory(
                start_pose,
                float(raw["path_radius"]),
                float(raw["reference_speed"]),
                float(raw["task_duration"]),
                float(raw["path_width"]),
            )
        if self.task_name == "figure8_tracking":
            return FigureEightTrajectory(
                start_pose,
                float(raw["path_width"]),
                float(raw["reference_speed"]),
                float(raw["task_duration"]),
                float(raw["path_width_tolerance"]),
            )
        raise ValueError(f"unknown task: {self.task_name}")

    def _parameter_vector(self) -> FloatArray:
        parameters = self.controller.parameters
        return np.array(
            [*parameters.damping, parameters.assist_gain, parameters.velocity_scale],
            dtype=np.float64,
        )

    def _apply_action(self, action: np.ndarray) -> float:
        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        before = self._parameter_vector()
        parameters = self.controller.parameters
        damping = parameters.damping.copy()
        damping[:2] += clipped_action[0] * self._action_scales[0]
        damping[2] += clipped_action[1] * self._action_scales[1]
        assist_gain = parameters.assist_gain + clipped_action[2] * self._action_scales[2]
        velocity_scale = parameters.velocity_scale + clipped_action[3] * self._action_scales[3]
        candidate = np.array([*damping, assist_gain, velocity_scale], dtype=np.float64)
        bounded = np.clip(candidate, self._parameter_lower, self._parameter_upper)
        self.controller.parameters = replace(
            parameters,
            damping=bounded[:3],
            assist_gain=float(bounded[3]),
            velocity_scale=float(bounded[4]),
        )
        return float(np.linalg.norm(bounded - before))

    def _tracking_error(self, reference_pose: FloatArray, actual_pose: FloatArray) -> FloatArray:
        error = reference_pose - actual_pose
        error[2] = wrap_angle(float(error[2]))
        return error

    def _observation(self, safety: bool = True) -> dict[str, np.ndarray]:
        assert self._task is not None
        reference_pose, _ = self._task.reference(self._elapsed_s)
        actual_pose = self.robot.end_effector_pose
        return {
            "joint_position": self.robot.qpos.astype(np.float32),
            "joint_velocity": self.robot.qvel.astype(np.float32),
            "end_effector_pose": actual_pose.astype(np.float32),
            "end_effector_velocity": self.robot.end_effector_velocity.astype(np.float32),
            "interaction_force": self._last_force.astype(np.float32),
            "reference_pose": reference_pose.astype(np.float32),
            "tracking_error": self._tracking_error(reference_pose, actual_pose).astype(np.float32),
            "admittance_parameters": self._parameter_vector().astype(np.float32),
            "task_progress": np.array([self._task.progress(self._elapsed_s)], dtype=np.float32),
            "safety_status": np.array([1.0 if safety else 0.0], dtype=np.float32),
        }

    def _unsafe_reason(
        self, pose: FloatArray, velocity: FloatArray, force: FloatArray
    ) -> str | None:
        if not np.all(np.isfinite(np.concatenate((pose, velocity, force)))):
            return "non_finite_state"
        if float(np.linalg.norm(force)) > float(self._simulation_limits["interaction_force_norm"]):
            return "interaction_force_limit"
        if float(np.linalg.norm(velocity)) > float(self._simulation_limits["task_speed_norm"]):
            return "task_speed_limit"
        if not self._workspace.contains(pose):
            return "workspace_limit"
        return None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset robot, controller, patient and trajectory state."""

        super().reset(seed=seed)
        if seed is not None:
            self._seed_value = int(seed)
        self.robot.reset(np.zeros(3), np.zeros(3))
        self.robot.set_joint_targets(np.zeros(3))
        start_pose = self.robot.end_effector_pose
        self._task = self._build_task(start_pose)
        self.controller = AdmittanceController(self._base_parameters, workspace=self._workspace)
        self.controller.reset(start_pose, np.zeros(3))
        self._patient = ImpedancePatient(
            self._profile,
            sample_time_s=self._base_parameters.sample_time_s,
            seed=self._seed_value,
        )
        reference_pose, reference_velocity = self._task.reference(0.0)
        self._patient.reset(reference_pose, reference_velocity)
        self._elapsed_s = 0.0
        self._last_progress = 0.0
        self._last_velocity = np.zeros(3, dtype=np.float64)
        self._last_force = np.zeros(3, dtype=np.float64)
        self._stagnation_s = 0.0
        self._episode_done = False
        self._last_reward_components = RewardComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return self._observation(), {
            "task_name": self.task_name,
            "patient_profile": self.patient_profile_name,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply one low-frequency parameter action and simulate one interval."""

        if self._episode_done:
            raise RuntimeError("step() called after episode termination; call reset()")
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (4,):
            raise ValueError(f"action must have shape (4,), got {action_array.shape}")
        parameter_change = self._apply_action(action_array)
        error_sum = 0.0
        force_sum = 0.0
        power_sum = 0.0
        assist_energy_sum = 0.0
        jerk_max = 0.0
        unsafe_reason: str | None = None
        for _ in range(self._physics_steps_per_action):
            assert self._task is not None
            reference_pose, reference_velocity = self._task.reference(self._elapsed_s)
            measured_pose = self.robot.end_effector_pose
            measured_velocity = self.robot.end_effector_velocity
            patient_output = self._patient.step(
                measured_pose,
                measured_velocity,
                reference_pose,
                reference_velocity,
            )
            admittance_output = self.controller.update(
                measured_pose,
                measured_velocity,
                reference_pose,
                reference_velocity,
                patient_output.force,
            )
            joint_target = self._ik.solve(admittance_output.desired_pose, self.robot.qpos)
            self.robot.set_joint_targets(joint_target)
            self.robot.set_external_wrench(patient_output.force)
            self.robot.step()
            actual_pose = self.robot.end_effector_pose
            actual_velocity = self.robot.end_effector_velocity
            error = self._tracking_error(reference_pose, actual_pose)
            error_sum += float(np.linalg.norm(error))
            force_sum += float(np.linalg.norm(patient_output.force))
            power_sum += patient_output.active_power_w
            assist_energy_sum += abs(float(np.dot(admittance_output.assist_force, actual_velocity)))
            jerk_max = max(
                jerk_max,
                float(np.linalg.norm((actual_velocity - self._last_velocity) / self._physics_dt)),
            )
            self._last_velocity[:] = actual_velocity
            self._last_force[:] = patient_output.force
            self._elapsed_s += self._physics_dt
            unsafe_reason = self._unsafe_reason(actual_pose, actual_velocity, patient_output.force)
            if unsafe_reason is not None:
                break

        assert self._task is not None
        progress = self._task.progress(self._elapsed_s)
        actual_pose = self.robot.end_effector_pose
        reference_pose, _ = self._task.reference(self._elapsed_s)
        tracking_error = self._tracking_error(reference_pose, actual_pose)
        average_error = error_sum / max(1, self._physics_steps_per_action)
        average_force = force_sum / max(1, self._physics_steps_per_action)
        average_power = power_sum / max(1, self._physics_steps_per_action)
        average_assistance = assist_energy_sum / max(1, self._physics_steps_per_action)
        progress_delta = max(0.0, progress - self._last_progress)
        if (
            progress_delta <= 1.0e-9
            and float(np.linalg.norm(self.robot.end_effector_velocity)) < 1.0e-4
        ):
            self._stagnation_s += self._action_interval_s
        else:
            self._stagnation_s = 0.0
        if unsafe_reason is None and self._stagnation_s >= float(
            self._simulation_limits["stagnation_seconds"]
        ):
            unsafe_reason = "stagnation"
        success = unsafe_reason is None and self._task.success(self._elapsed_s, tracking_error)
        terminated = success or unsafe_reason is not None
        truncated = not terminated and self._elapsed_s >= self._task.duration_s
        components = RewardComponents(
            progress=progress_delta,
            normalized_tracking_error=average_error / max(self._task.success_tolerance_m, 1.0e-6),
            excessive_force_penalty=max(
                0.0,
                average_force / float(self._simulation_limits["interaction_force_norm"]) - 0.5,
            ),
            motion_jerk_penalty=jerk_max / float(self._simulation_limits["task_acceleration_norm"]),
            robot_assistance_energy=average_assistance,
            parameter_change=parameter_change,
            positive_human_power=average_power,
            task_success=1.0 if success else 0.0,
            unsafe_termination=1.0 if unsafe_reason is not None else 0.0,
        )
        reward = components.weighted_total(self._reward_weights)
        self._last_progress = progress
        self._last_reward_components = components
        self._episode_done = terminated or truncated
        observation = self._observation(safety=unsafe_reason is None)
        info: dict[str, Any] = {
            "task_name": self.task_name,
            "patient_profile": self.patient_profile_name,
            "is_success": success,
            "unsafe_reason": unsafe_reason,
            "interaction_force_norm": average_force,
            "tracking_error_norm": float(np.linalg.norm(tracking_error)),
            "human_power_w": average_power,
            "parameter_change_norm": parameter_change,
            "reward_components": components.as_dict(),
            "task_progress": progress,
            "admittance_parameters": self._parameter_vector().tolist(),
        }
        return observation, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Phase 4 leaves rendering to the Phase 1 MuJoCo viewer."""

    def close(self) -> None:
        """Release environment resources."""
