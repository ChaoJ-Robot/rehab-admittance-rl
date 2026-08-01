"""Figure-eight tracking Gymnasium environment."""

from rehab_sim.envs.planar_rehab_env import PlanarRehabEnv


class Figure8TrackingEnv(PlanarRehabEnv):
    """Phase 4 figure-eight reference tracking task."""

    def __init__(
        self, patient_profile: str = "moderate", seed: int = 0, render_mode: str | None = None
    ) -> None:
        super().__init__("figure8_tracking", patient_profile, seed, render_mode)
