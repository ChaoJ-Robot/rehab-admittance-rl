"""Point-to-point Gymnasium environment."""

from rehab_sim.envs.planar_rehab_env import PlanarRehabEnv


class PointReachEnv(PlanarRehabEnv):
    """First Phase 4 point-to-point rehabilitation task."""

    def __init__(
        self, patient_profile: str = "moderate", seed: int = 0, render_mode: str | None = None
    ) -> None:
        super().__init__("point_to_point", patient_profile, seed, render_mode)
