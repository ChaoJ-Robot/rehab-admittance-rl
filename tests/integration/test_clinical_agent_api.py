"""Integration coverage for the clinical agent REST endpoints."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from backend.app.main import create_app  # noqa: E402
from backend.app.services.patient_store import PatientHistoryEntry, PatientStore  # noqa: E402


def _request(app, method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def _seed_patient(app, patient_id: str, completions: list[float], task: str = "point_to_point"):
    """Create a patient record with a fabricated training history."""

    session = app.state.session
    session.register_patient(patient_id, "moderate")
    record = session._patient_store.load(patient_id)
    assert record is not None
    for index, completion in enumerate(completions):
        record.history.append(
            PatientHistoryEntry(
                session_id=f"seed{index}",
                task=task,
                timestamp=time.time(),
                duration_s=4.0,
                score=completion * 2.0 - 1.0,
                completion_rate=completion,
                average_tracking_error=0.01,
            )
        )
    record.session_count = len(completions)
    session._patient_store.save(record)


def test_assessment_endpoint_reports_improving_trend(tmp_path: Path) -> None:
    app = create_app()
    app.state.session._patient_store = PatientStore(tmp_path)
    _seed_patient(app, "P010", [0.30, 0.50, 0.70, 0.85])

    response = _request(app, "GET", "/api/agent/assessment/P010")
    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"] == "improving"
    assert payload["sessions_analyzed"] == 4
    assert payload["avg_completion_recent"] == pytest.approx(0.68, abs=0.02)


def test_prescription_endpoint_upgrades_and_adoptable(tmp_path: Path) -> None:
    app = create_app()
    app.state.session._patient_store = PatientStore(tmp_path)
    _seed_patient(app, "P011", [0.70, 0.82, 0.88, 0.92])

    response = _request(app, "GET", "/api/agent/prescription/P011")
    assert response.status_code == 200
    payload = response.json()
    assert payload["difficulty_action"] == "upgrade"
    assert payload["task"] != "point_to_point"
    assert payload["rationale"]
    assert "reference_speed" in payload["task_params"]


def test_agent_endpoints_unknown_patient(tmp_path: Path) -> None:
    app = create_app()
    app.state.session._patient_store = PatientStore(tmp_path)
    assert _request(app, "GET", "/api/agent/assessment/ghost").status_code == 404
    assert _request(app, "GET", "/api/agent/prescription/ghost").status_code == 404
    assert _request(app, "GET", "/api/agent/assessment/bad%2Fid").status_code in (400, 404)
