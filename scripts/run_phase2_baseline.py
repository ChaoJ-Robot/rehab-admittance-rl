"""Run fixed-parameter admittance baseline force experiments.

The script closes the classical simulation loop as:

``wrench -> admittance task pose -> bounded IK target -> MuJoCo position loop``.

No RL policy or motor-torque command is used in this Phase 2 baseline.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from rehab_sim.config import load_yaml
from rehab_sim.controllers import AdmittanceController
from rehab_sim.robot import (
    DampedLeastSquaresIK,
    MujocoPlanarRobot,
    Planar3RGeometry,
    WorkspaceBounds,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _wrench_pattern(experiment: str, time_s: float) -> np.ndarray:
    """Return a deterministic test wrench ``[Fx,Fy,Tz]``."""

    if experiment == "step":
        return np.array([0.8 if time_s >= 0.5 else 0.0, 0.0, 0.0])
    if experiment == "reverse":
        if time_s < 0.5:
            fx = 0.0
        elif time_s < 1.5:
            fx = 0.8
        elif time_s < 2.5:
            fx = -0.8
        else:
            fx = 0.0
        return np.array([fx, 0.0, 0.0])
    if experiment == "sine":
        active_time = max(0.0, time_s - 0.5)
        return np.array(
            [
                0.6 * math.sin(2.0 * math.pi * 0.5 * active_time),
                0.35 * math.sin(2.0 * math.pi * 0.25 * active_time),
                0.08 * math.sin(2.0 * math.pi * 0.4 * active_time),
            ]
        )
    raise ValueError(f"unknown experiment: {experiment}")


def _load_components() -> tuple[MujocoPlanarRobot, AdmittanceController, DampedLeastSquaresIK]:
    root = _project_root()
    robot_config = load_yaml(root / "configs" / "robot.yaml")
    admittance_config = load_yaml(root / "configs" / "admittance.yaml")
    geometry = Planar3RGeometry.from_config(robot_config)
    workspace = WorkspaceBounds.from_config(robot_config)
    model_path = root / str(robot_config["robot"]["mujoco_model"])
    robot = MujocoPlanarRobot(model_path, geometry, workspace=workspace)
    controller = AdmittanceController.from_config(admittance_config, workspace=workspace)
    ik = DampedLeastSquaresIK(
        geometry=geometry,
        joint_lower=robot.joint_ranges[:, 0],
        joint_upper=robot.joint_ranges[:, 1],
    )
    return robot, controller, ik


def _run(experiment: str, duration_s: float) -> list[dict[str, float]]:
    robot, controller, ik = _load_components()
    dt = controller.parameters.sample_time_s
    steps = int(round(duration_s / dt))
    if steps < 1:
        raise ValueError("duration_s must contain at least one controller step")

    zero = np.zeros(3, dtype=np.float64)
    robot.reset(zero, zero)
    robot.set_joint_targets(zero)
    reference_pose = robot.end_effector_pose
    controller.reset(reference_pose, zero)
    rows: list[dict[str, float]] = []

    for step in range(steps):
        time_s = step * dt
        wrench = _wrench_pattern(experiment, time_s)
        measured_pose = robot.end_effector_pose
        measured_velocity = robot.end_effector_velocity
        output = controller.update(
            measured_pose=measured_pose,
            measured_velocity=measured_velocity,
            reference_pose=reference_pose,
            reference_velocity=zero,
            interaction_wrench=wrench,
        )
        joint_target = ik.solve(output.desired_pose, robot.qpos)
        robot.set_joint_targets(joint_target)
        robot.set_external_wrench(wrench)
        robot.step()
        actual_pose = robot.end_effector_pose
        actual_velocity = robot.end_effector_velocity
        row: dict[str, float] = {"time_s": time_s}
        for prefix, value in (
            ("wrench_raw", wrench),
            ("wrench_filtered", output.filtered_wrench),
            ("wrench_effective", output.effective_wrench),
            ("desired_pose", output.desired_pose),
            ("actual_pose", actual_pose),
            ("desired_velocity", output.desired_velocity),
            ("actual_velocity", actual_velocity),
            ("desired_acceleration", output.desired_acceleration),
            ("assist_force", output.assist_force),
            ("joint_target", joint_target),
        ):
            for axis, component in zip(("x", "y", "theta"), value, strict=True):
                row[f"{prefix}_{axis}"] = float(component)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _polyline(values: list[float], x: float, y: float, width: float, height: float) -> str:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        margin = max(1.0, abs(low) * 0.1)
        low -= margin
        high += margin
    points = []
    for index, value in enumerate(values):
        px = x + width * index / max(1, len(values) - 1)
        py = y + height * (high - value) / (high - low)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _write_svg(path: Path, rows: list[dict[str, float]], experiment: str) -> None:
    """Write dependency-free SVG force, velocity and position curves."""

    width, height = 1200, 900
    panels = (
        (
            "Interaction force",
            ("wrench_raw_x", "wrench_raw_y", "wrench_raw_theta"),
            ("Fx", "Fy", "Tz"),
        ),
        (
            "Task velocity",
            ("actual_velocity_x", "actual_velocity_y", "actual_velocity_theta"),
            ("vx", "vy", "omega"),
        ),
        (
            "Task position",
            ("actual_pose_x", "actual_pose_y", "actual_pose_theta"),
            ("x", "y", "theta"),
        ),
    )
    colors = ("#d1495b", "#00798c", "#edae49")
    x_values = [row["time_s"] for row in rows]
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f7f7"/>',
        f'<text x="40" y="30" font-family="sans-serif" font-size="20">'
        f"Phase 2 baseline: {html.escape(experiment)}</text>",
    ]
    for panel_index, (title, keys, labels) in enumerate(panels):
        x, y, panel_width, panel_height = 70, 60 + panel_index * 275, 1080, 210
        body.append(
            f'<rect x="{x}" y="{y}" width="{panel_width}" '
            f'height="{panel_height}" fill="white" stroke="#444"/>'
        )
        body.append(
            f'<text x="{x + 10}" y="{y + 22}" font-family="sans-serif" '
            f'font-size="16">{title}</text>'
        )
        for key, label, color in zip(keys, labels, colors, strict=True):
            values = [row[key] for row in rows]
            points = _polyline(values, x, y, panel_width, panel_height)
            body.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}"/>'
            )
            legend_x = x + 10 + 75 * labels.index(label)
            body.append(
                f'<text x="{legend_x}" y="{y + panel_height - 8}" fill="{color}" '
                f'font-family="sans-serif" font-size="13">{label}</text>'
            )
        body.append(
            f'<text x="{x + panel_width - 55}" y="{y + panel_height + 18}" '
            f'font-family="sans-serif" font-size="12">{x_values[-1]:.2f}s</text>'
        )
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def _write_summary(path: Path, rows: list[dict[str, float]], experiment: str) -> None:
    initial = np.array([rows[0][f"actual_pose_{axis}"] for axis in ("x", "y", "theta")])
    final = np.array([rows[-1][f"actual_pose_{axis}"] for axis in ("x", "y", "theta")])
    max_velocity = max(
        float(np.linalg.norm([row[f"actual_velocity_{axis}"] for axis in ("x", "y", "theta")]))
        for row in rows
    )
    max_force = max(
        float(np.linalg.norm([row[f"wrench_raw_{axis}"] for axis in ("x", "y", "theta")]))
        for row in rows
    )
    summary: dict[str, Any] = {
        "experiment": experiment,
        "sample_count": len(rows),
        "duration_s": rows[-1]["time_s"],
        "initial_pose": initial.tolist(),
        "final_pose": final.tolist(),
        "final_drift_norm": float(np.linalg.norm(final - initial)),
        "max_task_velocity_norm": max_velocity,
        "max_wrench_norm": max_force,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    """Run one deterministic Phase 2 baseline experiment."""

    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("step", "sine", "reverse"), default="step")
    parser.add_argument(
        "--duration", type=float, default=3.0, help="experiment duration in seconds"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiments" / "reports" / "phase2_baseline",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")

    rows = _run(args.experiment, args.duration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.experiment}_baseline"
    _write_csv(args.output_dir / f"{stem}.csv", rows)
    _write_svg(args.output_dir / f"{stem}.svg", rows, args.experiment)
    _write_summary(args.output_dir / f"{stem}.json", rows, args.experiment)
    print(f"wrote {len(rows)} samples to {args.output_dir / stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
