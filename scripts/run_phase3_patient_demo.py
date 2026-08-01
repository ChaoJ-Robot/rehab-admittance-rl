"""Generate open-loop virtual-patient force and fatigue curves.

This is a Phase 3 force-generator demonstration, not a Gymnasium environment
and not a robot controller. The prescribed current trajectory provides the
patient model with a deterministic tracking error and velocity signal.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
from rehab_sim.config import load_yaml
from rehab_sim.patients import ImpedancePatient, load_patient_profiles


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _trajectory(time_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth reference pose and velocity for the demo."""

    x = 0.02 * time_s
    y = 0.05 * math.sin(2.0 * math.pi * 0.2 * time_s)
    theta = 0.10 * math.sin(2.0 * math.pi * 0.15 * time_s)
    velocity = np.array(
        [
            0.02,
            0.05 * 2.0 * math.pi * 0.2 * math.cos(2.0 * math.pi * 0.2 * time_s),
            0.10 * 2.0 * math.pi * 0.15 * math.cos(2.0 * math.pi * 0.15 * time_s),
        ]
    )
    return np.array([x, y, theta]), velocity


def _run_profile(name: str, patient: ImpedancePatient, duration_s: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    steps = int(round(duration_s / patient.sample_time_s))
    for index in range(steps):
        time_s = index * patient.sample_time_s
        reference, reference_velocity = _trajectory(time_s)
        actual_pose = 0.55 * reference
        actual_velocity = 0.55 * reference_velocity
        output = patient.step(actual_pose, actual_velocity, reference, reference_velocity)
        row = {
            "time_s": time_s,
            "fatigue": output.fatigue,
            "active_power_w": output.active_power_w,
            "effective_strength": output.effective_strength,
        }
        for prefix, value in (
            ("force", output.force),
            ("active_force", output.active_force),
            ("noise_force", output.noise_force),
            ("tremor_force", output.tremor_force),
            ("reference_pose", reference),
            ("actual_pose", actual_pose),
        ):
            for axis, component in zip(("x", "y", "theta"), value, strict=True):
                row[f"{prefix}_{axis}"] = float(component)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _polyline(values: list[float], x: float, y: float, width: float, height: float) -> str:
    low, high = min(values), max(values)
    if math.isclose(low, high):
        margin = max(1.0, abs(low) * 0.1)
        low -= margin
        high += margin
    return " ".join(
        f"{x + width * i / max(1, len(values) - 1):.2f},"
        f"{y + height * (high - value) / (high - low):.2f}"
        for i, value in enumerate(values)
    )


def _write_svg(path: Path, name: str, rows: list[dict[str, float]]) -> None:
    width, height = 1000, 760
    panels = (
        ("Fatigue", "fatigue", "#d1495b"),
        ("Active power (W)", "active_power_w", "#00798c"),
        ("Patient force Fx (N)", "force_x", "#edae49"),
    )
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f7f7f7"/>',
        f'<text x="35" y="30" font-family="sans-serif" font-size="20">'
        f"Patient: {html.escape(name)}</text>",
    ]
    for index, (title, key, color) in enumerate(panels):
        x, y, panel_width, panel_height = 60, 55 + index * 230, 890, 175
        values = [row[key] for row in rows]
        points = _polyline(values, x, y, panel_width, panel_height)
        body.extend(
            [
                f'<rect x="{x}" y="{y}" width="{panel_width}" '
                f'height="{panel_height}" fill="white" stroke="#444"/>',
                f'<text x="{x + 10}" y="{y + 22}" font-family="sans-serif" '
                f'font-size="16">{title}</text>',
                f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}"/>',
            ]
        )
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> int:
    """Run all configured profiles with a fixed random seed."""

    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--sample-time", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiments" / "reports" / "phase3_patients",
    )
    args = parser.parse_args()
    if args.duration <= 0.0 or args.sample_time <= 0.0:
        parser.error("duration and sample-time must be positive")

    profiles = load_patient_profiles(load_yaml(root / "configs" / "patient_profiles.yaml"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float]] = {}
    for index, (name, profile) in enumerate(sorted(profiles.items())):
        patient = ImpedancePatient(profile, args.sample_time, seed=args.seed + index)
        rows = _run_profile(name, patient, args.duration)
        _write_csv(args.output_dir / f"{name}.csv", rows)
        _write_svg(args.output_dir / f"{name}.svg", name, rows)
        summary[name] = {
            "final_fatigue": rows[-1]["fatigue"],
            "max_active_power_w": max(row["active_power_w"] for row in rows),
            "mean_force_norm": float(
                np.mean(
                    [
                        np.linalg.norm([row[f"force_{axis}"] for axis in ("x", "y", "theta")])
                        for row in rows
                    ]
                )
            ),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote patient curves for {len(profiles)} profiles to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
