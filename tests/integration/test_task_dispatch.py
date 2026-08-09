"""Integration coverage for the dispatch closed loop (doctor -> patient -> data)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from backend.app.main import create_app  # noqa: E402
from backend.app.schemas.models import StartRequest  # noqa: E402
from backend.app.services.patient_store import PatientStore  # noqa: E402


def _app(tmp_path: Path):
    app = create_app()
    app.state.session._patient_store = PatientStore(tmp_path)
    return app


def _request(app, method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_dispatch_and_query_assignment(tmp_path: Path) -> None:
    app = _app(tmp_path)
    assert (
        _request(app, "POST", "/api/patients/P020", json={"profile": "moderate"}).status_code == 200
    )
    response = _request(
        app,
        "POST",
        "/api/patients/P020/assignments",
        json={
            "task": "circle_tracking",
            "task_params": {"reference_speed": 0.08},
            "due_date": "2026-08-15",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["task_params"]["reference_speed"] == 0.08

    listed = _request(app, "GET", "/api/patients/P020/assignments")
    assert listed.status_code == 200
    assert [item["assignment_id"] for item in listed.json()] == [payload["assignment_id"]]


def test_completing_session_settles_assignment(tmp_path: Path) -> None:
    app = _app(tmp_path)
    session = app.state.session
    session.register_patient("P021", "moderate")
    assignment = session.add_assignment("P021", "point_to_point", {}, "2026-08-20")
    assert assignment is not None

    request = StartRequest(
        task="point_to_point",
        patient_id="P021",
        mode="fixed",
        duration_s=4.0,
        assignment_id=assignment.assignment_id,
    )
    asyncio.run(session.start(request))
    session._cancel_run_task()
    for _ in range(int(10 * session.refresh_hz)):
        if session._state != "running":
            break
        session._tick()
    assert session._state in ("completed", "stopped")

    items = session.patient_assignments("P021")
    assert items is not None
    assert items[0].status == "completed"
    assert items[0].completed_session is not None


def test_assignment_endpoints_reject_unknown_patient(tmp_path: Path) -> None:
    app = _app(tmp_path)
    assert _request(app, "GET", "/api/patients/ghost/assignments").status_code == 404
    assert (
        _request(
            app, "POST", "/api/patients/ghost/assignments", json={"task": "maze_navigation"}
        ).status_code
        == 404
    )
    assert _request(app, "GET", "/api/patients/bad%2Fid/assignments").status_code in (400, 404)


def test_task_duration_param_overrides_config_default(tmp_path: Path) -> None:
    app = _app(tmp_path)
    session = app.state.session
    session.register_patient("P022", "moderate")
    request = StartRequest(
        task="point_to_point",
        patient_id="P022",
        mode="fixed",
        task_params={"task_duration": 2.0},
    )
    asyncio.run(session.start(request))
    session._cancel_run_task()
    assert session._duration_s == pytest.approx(2.0)
    for _ in range(int(4 * session.refresh_hz)):
        if session._state != "running":
            break
        session._tick()
    assert session._state == "completed"
