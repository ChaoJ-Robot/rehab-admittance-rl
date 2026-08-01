"""Evaluate a saved Phase 5 SAC model with its VecNormalize statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rehab_sim.rl.config import load_sac_config
from rehab_sim.rl.evaluation import evaluate_model
from rehab_sim.rl.vector_env import make_normalized_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize


def main() -> None:
    """Load a model/checkpoint and write deterministic evaluation metrics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="SB3 .zip model path")
    parser.add_argument("--vecnormalize", type=Path, required=True, help="VecNormalize .pkl path")
    parser.add_argument("--config", type=Path, default=Path("configs/rl_sac.yaml"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_sac_config(config_path)
    episodes = args.episodes or config.evaluation_episodes
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    model_path = args.model if args.model.is_absolute() else root / args.model
    vec_path = args.vecnormalize if args.vecnormalize.is_absolute() else root / args.vecnormalize

    placeholder_env = make_normalized_env(config, seed=300_000, n_envs=1, training=False)
    eval_env = VecNormalize.load(str(vec_path), placeholder_env.venv)
    eval_env.training = False
    eval_env.norm_reward = False
    model = SAC.load(str(model_path), env=eval_env, device=args.device)
    summary = evaluate_model(model, eval_env, episodes)
    payload = {
        "model": str(model_path),
        "vecnormalize": str(vec_path),
        "task_name": config.task_name,
        "patient_profile": config.patient_profile,
        "evaluation": summary.as_dict(),
    }
    output_path = args.output
    if output_path is not None:
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    eval_env.close()


if __name__ == "__main__":
    main()
