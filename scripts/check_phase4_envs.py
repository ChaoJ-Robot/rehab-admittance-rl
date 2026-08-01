"""Run the Phase 4 Gymnasium and random-policy smoke checks."""

from __future__ import annotations

import numpy as np
from rehab_sim.envs import CircleTrackingEnv, Figure8TrackingEnv, PointReachEnv
from stable_baselines3.common.env_checker import check_env


def main() -> None:
    """Validate all supported tasks and execute a short random rollout."""

    environment_types = (PointReachEnv, CircleTrackingEnv, Figure8TrackingEnv)
    for environment_type in environment_types:
        environment = environment_type(patient_profile="moderate", seed=42)
        check_env(environment, warn=True, skip_render_check=True)
        observation, _ = environment.reset(seed=42)
        steps = 0
        total_reward = 0.0
        while steps < 20:
            observation, reward, terminated, truncated, _ = environment.step(
                environment.action_space.sample()
            )
            if not all(np.all(np.isfinite(value)) for value in observation.values()):
                raise RuntimeError(f"non-finite observation in {environment_type.__name__}")
            if not np.isfinite(reward):
                raise RuntimeError(f"non-finite reward in {environment_type.__name__}")
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        environment.close()
        print(
            f"{environment_type.__name__}: check_env=ok, "
            f"random_steps={steps}, total_reward={total_reward:.4f}"
        )


if __name__ == "__main__":
    main()
