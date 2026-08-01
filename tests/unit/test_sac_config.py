from pathlib import Path

import pytest
from rehab_sim.rl.config import load_sac_config


def test_phase5_sac_config_is_numeric_and_multiseed() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_sac_config(root / "configs" / "rl_sac.yaml")
    assert config.total_timesteps > 0
    assert len(config.random_seeds) >= 5
    assert config.normalize_observation
    assert config.normalize_reward
    assert config.task_name == "point_to_point"
    assert config.policy_net_arch


def test_phase5_config_rejects_empty_seed_list(tmp_path: Path) -> None:
    config_path = tmp_path / "rl_sac.yaml"
    config_path.write_text(
        """
training:
  total_timesteps: 10
  random_seeds: []
  checkpoint_frequency: 5
  evaluation_frequency: 5
  evaluation_episodes: 1
environment:
  task: point_to_point
  patient_profile: moderate
  n_envs: 1
normalization:
  normalize_observation: true
  normalize_reward: true
  clip_observation: 10
  clip_reward: 10
sac:
  learning_rate: 0.001
  buffer_size: 32
  learning_starts: 1
  batch_size: 2
  tau: 0.005
  gamma: 0.99
  train_frequency: 1
  gradient_steps: 1
  ent_coef: auto
  policy_net_arch: [16]
  device: cpu
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="random_seeds"):
        load_sac_config(config_path)
