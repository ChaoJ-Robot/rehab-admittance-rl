"""Run Phase 10 comparisons and generate data, plots, report and optional video."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from rehab_sim.experiments.comparison import run_comparison
from rehab_sim.experiments.config import (
    VALID_METHODS,
    VALID_PATIENTS,
    VALID_TASKS,
    load_phase10_config,
)
from rehab_sim.experiments.video import record_demo_video


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return parsed


def _parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in _parse_csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase10.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task", choices=VALID_TASKS, default=None)
    parser.add_argument("--patients", type=_parse_csv, default=None)
    parser.add_argument("--methods", type=_parse_csv, default=None)
    parser.add_argument("--seeds", type=_parse_int_csv, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--train-timesteps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--quick", action="store_true", help="use one seed/episode and short RL runs"
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--video-method",
        choices=("fixed_admittance", "rule_adaptive", "fuzzy_control"),
        default="rule_adaptive",
    )
    parser.add_argument("--video-patient", choices=VALID_PATIENTS, default="moderate")
    return parser


def main() -> None:
    """Run the configured Phase 10 matrix."""

    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_phase10_config(config_path)
    if args.task is not None:
        config = replace(config, task_name=args.task)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.methods is not None:
        unknown = set(args.methods) - set(VALID_METHODS)
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
    if args.patients is not None:
        unknown = set(args.patients) - set(VALID_PATIENTS)
        if unknown:
            raise ValueError(f"unknown patients: {sorted(unknown)}")
    episodes = args.episodes
    seeds = args.seeds
    timesteps = args.train_timesteps
    if args.quick:
        episodes = 1
        seeds = (0,)
        timesteps = 32 if timesteps is None else timesteps
    if episodes is not None and episodes <= 0:
        raise ValueError("episodes must be positive")
    if timesteps is not None and timesteps < 0:
        raise ValueError("train-timesteps must be non-negative")
    output_dir = args.output_dir or config.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = run_comparison(
        root=root,
        config=config,
        output_dir=output_dir,
        methods=args.methods,
        patients=args.patients,
        seeds=seeds,
        episodes_per_condition=episodes,
        training_timesteps=timesteps,
    )
    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["command"] = sys.argv
    metadata["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    video_path = None
    if args.record_video:
        video_path = record_demo_video(
            output_path=output_dir / f"demo_{args.video_method}_{args.video_patient}.mp4",
            task_name=config.task_name,
            patient_profile=args.video_patient,
            method=args.video_method,
            seed=(seeds or config.seeds)[0],
            policy_parameters=config.policy_parameters,
            fps=config.video_fps,
            max_frames=config.video_max_frames,
        )
    result = {
        "output_dir": str(output_dir),
        "episode_rows": len(rows),
        "summary_conditions": len(summary),
        "video": str(video_path) if video_path is not None else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
