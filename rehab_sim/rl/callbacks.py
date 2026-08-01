"""SAC checkpoint and automatic evaluation callbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class MetadataCheckpointCallback(CheckpointCallback):
    """Save SB3 checkpoints together with reproducibility metadata."""

    def __init__(self, *args: Any, metadata: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._metadata = dict(metadata)

    def _on_step(self) -> bool:
        should_save = self.n_calls % self.save_freq == 0
        result = super()._on_step()
        if should_save:
            model_path = Path(self._checkpoint_path(extension="zip"))
            metadata = {
                **self._metadata,
                "artifact_type": "checkpoint",
                "checkpoint_timestep": self.num_timesteps,
                "model_file": model_path.name,
            }
            _write_json(model_path.with_suffix(".metadata.json"), metadata)
        return result


class EvaluationHistoryCallback(EvalCallback):
    """Run periodic deterministic evaluation and persist a JSON history."""

    def __init__(
        self,
        *args: Any,
        history_path: str | Path,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._history_path = Path(history_path)
        self._metadata = dict(metadata or {})

    def _on_step(self) -> bool:
        previous_best = self.best_mean_reward
        result = super()._on_step()
        if self.best_mean_reward > previous_best and self.best_model_save_path is not None:
            best_path = Path(self.best_model_save_path) / "best_model.zip"
            _write_json(
                best_path.with_suffix(".metadata.json"),
                {
                    **self._metadata,
                    "artifact_type": "best_model",
                    "evaluation_timestep": self.num_timesteps,
                    "model_file": best_path.name,
                },
            )
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if not self.evaluations_timesteps or not self.evaluations_results:
                return result
            index = len(self.evaluations_timesteps) - 1
            rewards = np.asarray(self.evaluations_results[index], dtype=np.float64)
            successes = None
            if self.evaluations_successes:
                successes = np.asarray(self.evaluations_successes[index], dtype=np.float64)
            history: list[dict[str, Any]] = []
            if self._history_path.exists():
                try:
                    loaded = json.loads(self._history_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        history = loaded
                except json.JSONDecodeError:
                    history = []
            entry: dict[str, Any] = {
                "timesteps": int(self.evaluations_timesteps[index]),
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "episode_length_mean": float(
                    np.mean(np.asarray(self.evaluations_length[index], dtype=np.float64))
                ),
            }
            if successes is not None:
                entry["success_rate"] = float(np.mean(successes))
            history.append(entry)
            _write_json(self._history_path, history)
        return result
