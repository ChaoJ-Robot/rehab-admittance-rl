import numpy as np
from rehab_sim.rl.evaluation import _EpisodeAccumulator


def test_evaluation_accumulates_force_and_parameter_oscillation() -> None:
    accumulator = _EpisodeAccumulator()
    for parameters, force in [
        ([3.0, 3.0, 0.25, 0.0, 1.0], 0.2),
        ([3.1, 3.0, 0.25, 0.0, 1.0], 0.4),
        ([3.0, 3.0, 0.25, 0.0, 1.0], 0.3),
    ]:
        accumulator.add_step(
            1.0,
            {
                "interaction_force_norm": force,
                "admittance_parameters": parameters,
                "is_success": False,
                "unsafe_reason": None,
            },
        )
    accumulator.finish({"is_success": True, "unsafe_reason": None})
    summary = accumulator.as_dict()
    assert summary["success"] is True
    assert summary["max_interaction_force"] == 0.4
    assert summary["mean_interaction_force"] == np.mean([0.2, 0.4, 0.3])
    assert summary["mean_parameter_change"] > 0.0
    assert summary["parameter_oscillation_rate"] > 0.0
