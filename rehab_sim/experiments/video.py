"""Headless MuJoCo demonstration-video generation for Phase 10."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from rehab_sim.envs import PointReachEnv
from rehab_sim.experiments.comparison import classical_policy


def record_demo_video(
    *,
    output_path: Path,
    task_name: str,
    patient_profile: str,
    method: str,
    seed: int,
    policy_parameters: Mapping[str, Mapping[str, float]],
    fps: int,
    max_frames: int,
) -> Path:
    """Record a robot/target trajectory as an MP4 using the headless renderer."""

    if task_name != "point_to_point":
        raise ValueError("Phase 10 video currently uses the point_to_point task")
    if fps <= 0 or max_frames <= 0:
        raise ValueError("fps and max_frames must be positive")
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    environment = PointReachEnv(patient_profile=patient_profile, seed=seed)
    policy = classical_policy(method, policy_parameters)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", macro_block_size=1)
    try:
        observation, _ = environment.reset(seed=seed)
        for _ in range(max_frames):
            figure, axis = plt.subplots(figsize=(7, 5))
            joint1, joint2, joint3 = environment._geometry.joint_positions(environment.robot.qpos)
            tip = environment.robot.end_effector_pose[:2]
            points = np.vstack(
                (
                    environment._geometry.joint1_origin,
                    joint2,
                    joint3,
                    tip,
                )
            )
            axis.plot(points[:, 0], points[:, 1], "o-", linewidth=4, label="robot")
            axis.scatter(
                observation["reference_pose"][0],
                observation["reference_pose"][1],
                marker="*",
                s=140,
                label="reference",
            )
            axis.scatter(tip[0], tip[1], marker="o", s=55, label="tool tip")
            axis.set_xlim(*environment._workspace.x)
            axis.set_ylim(*environment._workspace.y)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, alpha=0.3)
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
            axis.set_title(
                f"{method} | patient={patient_profile} | t={environment._elapsed_s:.1f}s"
            )
            axis.legend(loc="upper left")
            figure.tight_layout()
            figure.canvas.draw()
            frame = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)
            writer.append_data(frame[:, :, :3])
            plt.close(figure)
            observation, _, terminated, truncated, _ = environment.step(policy(observation))
            if terminated or truncated:
                break
    finally:
        writer.close()
        environment.close()
    return output_path
