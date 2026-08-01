from __future__ import annotations

import pytest


def test_phase9_generated_interfaces_are_available() -> None:
    pytest.importorskip("rehab_interfaces")
    from rehab_interfaces.msg import (  # noqa: PLC0415
        AdmittanceParameters,
        PolicyAction,
        SafetyState,
    )
    from rehab_interfaces.srv import EnableRl, StartTask  # noqa: PLC0415

    parameters = AdmittanceParameters()
    parameters.damping = [3.0, 3.0, 0.25]
    parameters.velocity_scale = 0.1
    action = PolicyAction()
    action.raw_action = [0.0, 0.0, 0.0, 0.0]
    safety = SafetyState()
    safety.reasons = ["test"]
    assert len(parameters.damping) == 3
    assert len(action.raw_action) == 4
    assert safety.reasons == ["test"]
    assert EnableRl.Request().enabled is False
    assert StartTask.Response().accepted is False
