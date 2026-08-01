from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
check_env = pytest.importorskip("stable_baselines3.common.env_checker").check_env

from rehab_sim.envs import CircleTrackingEnv, Figure8TrackingEnv, PointReachEnv  # noqa: E402


@pytest.mark.parametrize("env_type", [PointReachEnv, CircleTrackingEnv, Figure8TrackingEnv])
def test_stable_baselines_check_env(env_type) -> None:
    env = env_type(patient_profile="moderate", seed=5)
    check_env(env, warn=True, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_type", [PointReachEnv, CircleTrackingEnv, Figure8TrackingEnv])
def test_random_policy_runs_without_nan(env_type) -> None:
    env = env_type(patient_profile="mild", seed=6)
    observation, _ = env.reset(seed=6)
    for _ in range(10):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert all(np.all(np.isfinite(value)) for value in observation.values())
        assert np.isfinite(reward)
        assert "reward_components" in info
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()


def test_same_seed_and_action_are_reproducible() -> None:
    env_a = PointReachEnv(patient_profile="moderate", seed=9)
    env_b = PointReachEnv(patient_profile="moderate", seed=9)
    observation_a, _ = env_a.reset(seed=9)
    observation_b, _ = env_b.reset(seed=9)
    for key in observation_a:
        np.testing.assert_array_equal(observation_a[key], observation_b[key])
    action = np.array([0.25, -0.1, 0.0, 0.2], dtype=np.float32)
    next_a = env_a.step(action)
    next_b = env_b.step(action)
    for key in next_a[0]:
        np.testing.assert_allclose(next_a[0][key], next_b[0][key], atol=1.0e-7)
    assert next_a[1:] == next_b[1:]
    env_a.close()
    env_b.close()
