"""MuJoCo runtime adapter for the planar rehabilitation robot.

This Phase 1 adapter owns only robot-state access, joint target commands,
end-effector wrench injection, stepping and model-level limits. It does not
implement admittance control, RL, safety projection or fallback policy.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from rehab_sim.robot.kinematics import Planar3RGeometry, WorkspaceBounds, _vector

FloatArray = NDArray[np.float64]


class NumericalSimulationError(RuntimeError):
    """Raised when MuJoCo produces a non-finite state."""


class MujocoPlanarRobot:
    """Load and step the configured three-joint planar robot model."""

    JOINT_NAMES = ("joint1", "joint2", "joint3")
    ACTUATOR_NAMES = ("joint1_target", "joint2_target", "joint3_target")

    def __init__(
        self,
        model_path: str | Path,
        geometry: Planar3RGeometry,
        workspace: WorkspaceBounds | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.geometry = geometry
        self.workspace = workspace
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array(
            [self.model.joint(name).id for name in self.JOINT_NAMES], dtype=np.int32
        )
        self.actuator_ids = np.array(
            [self.model.actuator(name).id for name in self.ACTUATOR_NAMES], dtype=np.int32
        )
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_addresses = self.model.jnt_dofadr[self.joint_ids]
        self.site_id = self.model.site("tool_tip").id
        self.site_body_id = int(self.model.site_bodyid[self.site_id])
        self.joint_ranges = self.model.jnt_range[self.joint_ids].copy()
        self.control_ranges = self.model.actuator_ctrlrange[self.actuator_ids].copy()
        self._external_wrench = np.zeros(3, dtype=np.float64)
        self.reset()

    @property
    def qpos(self) -> FloatArray:
        """Current joint positions in radians, ordered as joint1..joint3."""

        return self.data.qpos[self.qpos_addresses].copy()

    @property
    def qvel(self) -> FloatArray:
        """Current joint velocities in rad/s, ordered as joint1..joint3."""

        return self.data.qvel[self.qvel_addresses].copy()

    @property
    def end_effector_pose(self) -> FloatArray:
        """Current site pose ``[x, y, theta]`` in m, m, rad."""

        position = self.data.site_xpos[self.site_id]
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        theta = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        return np.array([position[0], position[1], theta], dtype=np.float64)

    @property
    def end_effector_velocity(self) -> FloatArray:
        """Current task velocity ``[vx, vy, omega]`` in SI units."""

        return self.geometry.jacobian(self.qpos) @ self.qvel

    @property
    def external_wrench(self) -> FloatArray:
        """Configured interaction wrench ``[Fx, Fy, Tz]`` in N, N, N*m."""

        return self._external_wrench.copy()

    def reset(self, qpos: ArrayLike | None = None, qvel: ArrayLike | None = None) -> None:
        """Reset state and targets to zero or the supplied joint state."""

        position = np.zeros(3, dtype=np.float64) if qpos is None else _vector(qpos, 3, "qpos")
        velocity = np.zeros(3, dtype=np.float64) if qvel is None else _vector(qvel, 3, "qvel")
        self.data.qpos[self.qpos_addresses] = position
        self.data.qvel[self.qvel_addresses] = velocity
        self.data.ctrl[self.actuator_ids] = np.clip(
            position, self.control_ranges[:, 0], self.control_ranges[:, 1]
        )
        self._external_wrench[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_joint_targets(self, targets: ArrayLike) -> FloatArray:
        """Set independent position targets, clipped by model actuator limits.

        If workspace bounds were supplied, the predicted target pose is
        rejected before it reaches the actuator array.
        """

        requested = _vector(targets, 3, "joint targets")
        clipped = np.clip(requested, self.control_ranges[:, 0], self.control_ranges[:, 1])
        if self.workspace is not None and not self.workspace.contains(
            self.geometry.forward(clipped)
        ):
            raise ValueError("joint target predicts a pose outside the configured workspace")
        self.data.ctrl[self.actuator_ids] = clipped
        return clipped.copy()

    def set_joint_state(self, qpos: ArrayLike, qvel: ArrayLike | None = None) -> None:
        """Set a test state and run ``mj_forward`` without stepping dynamics."""

        position = _vector(qpos, 3, "qpos")
        velocity = np.zeros(3, dtype=np.float64) if qvel is None else _vector(qvel, 3, "qvel")
        self.data.qpos[self.qpos_addresses] = position
        self.data.qvel[self.qvel_addresses] = velocity
        mujoco.mj_forward(self.model, self.data)

    def set_external_wrench(self, wrench: ArrayLike) -> None:
        """Apply ``[Fx,Fy,Tz]`` at the end-effector site in world coordinates."""

        self._external_wrench[:] = _vector(wrench, 3, "external wrench")
        self._refresh_external_wrench()

    def clear_external_wrench(self) -> None:
        """Remove the configured end-effector interaction wrench."""

        self._external_wrench[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

    def _refresh_external_wrench(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        force = np.array([self._external_wrench[0], self._external_wrench[1], 0.0])
        torque = np.array([0.0, 0.0, self._external_wrench[2]])
        mujoco.mj_applyFT(
            self.model,
            self.data,
            force,
            torque,
            self.data.site_xpos[self.site_id],
            self.site_body_id,
            self.data.qfrc_applied,
        )

    def step(self, steps: int = 1) -> None:
        """Advance the model by ``steps`` MuJoCo timesteps."""

        if steps < 1:
            raise ValueError("steps must be at least 1")
        for _ in range(steps):
            self._refresh_external_wrench()
            mujoco.mj_step(self.model, self.data)
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                raise NumericalSimulationError("MuJoCo state became non-finite")

    def within_joint_limits(self, qpos: ArrayLike | None = None) -> bool:
        """Return whether all joint positions are within MJCF position limits."""

        value = self.qpos if qpos is None else _vector(qpos, 3, "qpos")
        return bool(
            np.all(value >= self.joint_ranges[:, 0]) and np.all(value <= self.joint_ranges[:, 1])
        )

    def within_workspace(self, pose: ArrayLike | None = None) -> bool:
        """Return whether a task pose is inside the configured workspace bounds."""

        if self.workspace is None:
            raise RuntimeError("workspace bounds were not configured")
        value = self.end_effector_pose if pose is None else _vector(pose, 3, "pose")
        return self.workspace.contains(value)
