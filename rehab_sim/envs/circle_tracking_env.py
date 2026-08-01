"""Circular tracking Gymnasium environment."""

from rehab_sim.envs.planar_rehab_env import PlanarRehabEnv


class CircleTrackingEnv(PlanarRehabEnv):
    """Phase 4 circular reference tracking task."""

    def __init__(
        self, patient_profile: str = "moderate", seed: int = 0, render_mode: str | None = None
    ) -> None:
        super().__init__("circle_tracking", patient_profile, seed, render_mode)
