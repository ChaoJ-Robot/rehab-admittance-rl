"""Train and evaluate Phase 5 SAC parameter-adaptation policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rehab_sim.rl.callbacks import EvaluationHistoryCallback, MetadataCheckpointCallback
from rehab_sim.rl.config import SACExperimentConfig, load_sac_config
from rehab_sim.rl.evaluation import EvaluationSummary, evaluate_model, evaluate_random_policy
from rehab_sim.rl.vector_env import make_normalized_env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import sync_envs_normalization


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _config_digest(root: Path) -> str:
    """Hash all YAML configuration files in deterministic path order."""

    digest = hashlib.sha256()
    for path in sorted((root / "configs").glob("*.yaml")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("seeds must contain at least one integer")
    return seeds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rl_sac.yaml"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/trained_models/phase5_sac")
    )
    parser.add_argument("--run-name", default=None, help="Output run name; defaults to UTC time")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=None, help="Comma-separated seeds")
    parser.add_argument("--checkpoint-frequency", type=int, default=None)
    parser.add_argument("--evaluation-frequency", type=int, default=None)
    parser.add_argument(
        "--task", default=None, choices=["point_to_point", "circle_tracking", "figure8_tracking"]
    )
    parser.add_argument("--patient-profile", default=None, choices=["mild", "moderate", "severe"])
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--device", default=None, help="SB3 device, for example cpu or auto")
    return parser


def _apply_overrides(config: SACExperimentConfig, args: argparse.Namespace) -> SACExperimentConfig:
    updates: dict[str, Any] = {}
    argument_names = {
        "total_timesteps": "total_timesteps",
        "random_seeds": "seeds",
        "checkpoint_frequency": "checkpoint_frequency",
        "evaluation_frequency": "evaluation_frequency",
        "task_name": "task",
        "patient_profile": "patient_profile",
        "n_envs": "n_envs",
        "evaluation_episodes": "eval_episodes",
        "learning_starts": "learning_starts",
        "device": "device",
    }
    for name, argument_name in argument_names.items():
        value = getattr(args, argument_name, None)
        if value is not None:
            updates[name] = value
    if updates.get("total_timesteps", config.total_timesteps) <= 0:
        raise ValueError("total_timesteps must be positive")
    if updates.get("n_envs", config.n_envs) <= 0:
        raise ValueError("n_envs must be positive")
    if updates.get("evaluation_episodes", config.evaluation_episodes) <= 0:
        raise ValueError("eval-episodes must be positive")
    if updates.get("checkpoint_frequency", config.checkpoint_frequency) <= 0:
        raise ValueError("checkpoint-frequency must be positive")
    if updates.get("evaluation_frequency", config.evaluation_frequency) <= 0:
        raise ValueError("evaluation-frequency must be positive")
    if updates.get("learning_starts", config.learning_starts) < 0:
        raise ValueError("learning-starts must be non-negative")
    return replace(config, **updates)


def _summary_payload(
    policy: EvaluationSummary,
    random: EvaluationSummary,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"policy": policy.as_dict(), "random": random.as_dict()}
    payload["reward_improvement_over_random"] = policy.reward_mean - random.reward_mean
    payload["success_rate_improvement_over_random"] = policy.success_rate - random.success_rate
    return payload


def _train_seed(
    run_dir: Path,
    config: SACExperimentConfig,
    seed: int,
    config_path: Path,
    config_hash: str,
    git_commit: str,
) -> dict[str, Any]:
    seed_dir = run_dir / f"seed_{seed:04d}"
    checkpoint_dir = seed_dir / "checkpoints"
    eval_dir = seed_dir / "evaluation"
    seed_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "phase": 5,
        "algorithm": "SAC",
        "task_name": config.task_name,
        "patient_profile": config.patient_profile,
        "seed": seed,
        "config_file": str(config_path),
        "config_sha256": config_hash,
        "git_commit": git_commit,
        "command": sys.argv,
    }
    _write_json(seed_dir / "metadata.json", metadata)

    set_random_seed(seed)
    train_env = make_normalized_env(config, seed=seed, n_envs=config.n_envs, training=True)
    eval_env = make_normalized_env(config, seed=seed + 100_000, n_envs=1, training=False)
    callback = CallbackList(
        [
            MetadataCheckpointCallback(
                save_freq=max(1, config.checkpoint_frequency // config.n_envs),
                save_path=str(checkpoint_dir),
                name_prefix="sac_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
                metadata=metadata,
                verbose=1,
            ),
            EvaluationHistoryCallback(
                eval_env=eval_env,
                n_eval_episodes=config.evaluation_episodes,
                eval_freq=max(1, config.evaluation_frequency // config.n_envs),
                log_path=str(eval_dir),
                best_model_save_path=str(seed_dir / "best_model"),
                deterministic=True,
                history_path=seed_dir / "evaluation_history.json",
                metadata=metadata,
                verbose=1,
            ),
        ]
    )
    model = SAC(
        "MultiInputPolicy",
        train_env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=config.train_frequency,
        gradient_steps=config.gradient_steps,
        ent_coef=config.ent_coef,
        policy_kwargs={"net_arch": list(config.policy_net_arch)},
        tensorboard_log=str(seed_dir / "tensorboard"),
        device=config.device,
        seed=seed,
        verbose=1,
    )
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        tb_log_name=f"seed_{seed:04d}",
        progress_bar=False,
    )

    final_model_path = seed_dir / "final_model"
    vecnormalize_path = seed_dir / "vecnormalize.pkl"
    model.save(str(final_model_path))
    train_env.save(str(vecnormalize_path))
    _write_json(
        final_model_path.with_suffix(".metadata.json"),
        {**metadata, "artifact_type": "final_model", "model_file": "final_model.zip"},
    )
    _write_json(
        vecnormalize_path.with_suffix(".metadata.json"),
        {**metadata, "artifact_type": "vecnormalize", "model_file": vecnormalize_path.name},
    )

    sync_envs_normalization(train_env, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    loaded_model = SAC.load(str(final_model_path), env=eval_env, device=config.device)
    policy_summary = evaluate_model(loaded_model, eval_env, config.evaluation_episodes)
    random_summary = evaluate_random_policy(
        config.task_name,
        config.patient_profile,
        config.evaluation_episodes,
        seed + 100_000,
    )
    summary = {
        **metadata,
        "total_timesteps": config.total_timesteps,
        "evaluation": _summary_payload(policy_summary, random_summary),
    }
    _write_json(seed_dir / "evaluation_summary.json", summary)
    train_env.close()
    eval_env.close()
    return summary


def main() -> None:
    """Run one SAC experiment for every configured random seed."""

    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = _apply_overrides(load_sac_config(config_path), args)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    output_root = (
        (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    )
    run_dir = output_root / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "rl_sac.yaml")
    config_hash = _config_digest(root)
    git_commit = _git_commit(root)
    _write_json(
        run_dir / "run_metadata.json",
        {
            "phase": 5,
            "algorithm": "SAC",
            "git_commit": git_commit,
            "config_sha256": config_hash,
            "config": asdict(config),
            "command": sys.argv,
        },
    )
    summaries = [
        _train_seed(run_dir, config, seed, config_path, config_hash, git_commit)
        for seed in config.random_seeds
    ]
    policy_rewards = [
        float(summary["evaluation"]["policy"]["reward_mean"]) for summary in summaries
    ]
    random_rewards = [
        float(summary["evaluation"]["random"]["reward_mean"]) for summary in summaries
    ]
    policy_success = [
        float(summary["evaluation"]["policy"]["success_rate"]) for summary in summaries
    ]
    aggregate = {
        "seed_count": len(summaries),
        "policy_reward_mean": float(np.mean(policy_rewards)),
        "policy_reward_std": float(np.std(policy_rewards)),
        "random_reward_mean": float(np.mean(random_rewards)),
        "random_reward_std": float(np.std(random_rewards)),
        "reward_improvement_mean": float(np.mean(np.asarray(policy_rewards) - random_rewards)),
        "policy_success_rate_mean": float(np.mean(policy_success)),
    }
    _write_json(run_dir / "multi_seed_summary.json", {"seeds": summaries, "aggregate": aggregate})
    print(
        json.dumps({"run_dir": str(run_dir), "aggregate": aggregate}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
