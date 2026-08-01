from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from rehab_sim.rl.config import load_sac_config  # noqa: E402
from rehab_sim.rl.vector_env import make_normalized_env  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402


def test_sac_model_and_vecnormalize_can_be_saved_and_loaded(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_sac_config(root / "configs" / "rl_sac.yaml")
    train_env = make_normalized_env(config, seed=31, n_envs=1, training=True)
    model = SAC(
        "MultiInputPolicy",
        train_env,
        buffer_size=32,
        learning_starts=100,
        batch_size=2,
        policy_kwargs={"net_arch": [16]},
        device="cpu",
        seed=31,
    )
    model_path = tmp_path / "model"
    vec_path = tmp_path / "vecnormalize.pkl"
    model.save(str(model_path))
    train_env.save(str(vec_path))
    base_env = make_normalized_env(config, seed=32, n_envs=1, training=False)
    eval_env = VecNormalize.load(str(vec_path), base_env.venv)
    eval_env.training = False
    eval_env.norm_reward = False
    loaded = SAC.load(str(model_path), env=eval_env, device="cpu")
    observation = eval_env.reset()
    action, _ = loaded.predict(observation, deterministic=True)
    assert eval_env.action_space.contains(action[0])
    train_env.close()
    eval_env.close()
