"""Run the Phase 1 MuJoCo robot model with optional visualization."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from rehab_sim.config import load_yaml
from rehab_sim.robot import MujocoPlanarRobot, Planar3RGeometry


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _make_robot() -> MujocoPlanarRobot:
    root = _project_root()
    config = load_yaml(root / "configs" / "robot.yaml")
    robot_config = config["robot"]
    model_path = root / str(robot_config["mujoco_model"])
    geometry = Planar3RGeometry.from_config(config)
    return MujocoPlanarRobot(model_path, geometry)


def main() -> int:
    """Run a finite headless test or launch the MuJoCo passive viewer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="step without opening a GUI")
    parser.add_argument("--steps", type=int, default=1000, help="number of simulation steps")
    parser.add_argument(
        "--qpos",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("Q1", "Q2", "Q3"),
        help="initial joint position in radians",
    )
    parser.add_argument(
        "--wrench",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("FX", "FY", "TZ"),
        help="constant end-effector wrench in N, N, N*m",
    )
    parser.add_argument("--top", action="store_true", help="use a top-down camera")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    robot = _make_robot()
    qpos = np.asarray(args.qpos, dtype=np.float64)
    robot.reset(qpos=qpos, qvel=np.zeros(3))
    robot.set_joint_targets(qpos)
    robot.set_external_wrench(args.wrench)

    print(
        f"loaded {robot.model_path.name}: nq={robot.model.nq}, nv={robot.model.nv}, "
        f"nu={robot.model.nu}, nsite={robot.model.nsite}"
    )

    if args.headless:
        robot.step(args.steps)
        print("qpos:", robot.qpos.tolist())
        print("qvel:", robot.qvel.tolist())
        print("end_effector_pose:", robot.end_effector_pose.tolist())
        print("finite:", bool(np.all(np.isfinite(robot.data.qpos))))
        return 0

    from mujoco import viewer

    print("Viewer started; close the MuJoCo window to exit.")
    with viewer.launch_passive(robot.model, robot.data) as viewer_handle:
        viewer_handle.cam.lookat[:] = [-0.05, -0.18, -0.34]
        viewer_handle.cam.distance = 1.35
        viewer_handle.cam.azimuth = 0 if args.top else 135
        viewer_handle.cam.elevation = -88 if args.top else -32
        while viewer_handle.is_running():
            start = time.perf_counter()
            robot.step()
            viewer_handle.sync()
            time.sleep(max(0.0, robot.model.opt.timestep - (time.perf_counter() - start)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
