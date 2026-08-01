from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from backend.app.main import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def test_phase7_rest_controls_and_report() -> None:
    app = create_app()
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["mode"] == "simulation_only"

        config = client.get("/api/config")
        assert config.status_code == 200
        assert config.json()["refresh_hz"] == 20
        assert config.json()["hardware_validation_required"] is True

        started = client.post(
            "/api/session/start",
            json={
                "task": "point_to_point",
                "patient_profile": "moderate",
                "mode": "fixed",
                "duration_s": 0.2,
            },
        )
        assert started.status_code == 200
        assert started.json()["state"] == "running"

        paused = client.post("/api/session/pause")
        assert paused.status_code == 200
        assert paused.json()["state"] == "paused"
        resumed = client.post("/api/session/resume")
        assert resumed.status_code == 200
        assert resumed.json()["state"] == "running"
        mode = client.post("/api/session/mode", json={"mode": "rl"})
        assert mode.status_code == 200
        assert mode.json()["mode"] == "rl"

        time.sleep(0.08)
        stopped = client.post("/api/session/stop")
        assert stopped.status_code == 200
        assert stopped.json()["state"] == "stopped"
        report = client.get("/api/report")
        assert report.status_code == 200
        assert "average_tracking_error" in report.json()


def test_phase7_websocket_streams_telemetry() -> None:
    app = create_app()
    with TestClient(app) as client:
        started = client.post(
            "/api/session/start",
            json={
                "task": "circle_tracking",
                "patient_profile": "mild",
                "mode": "rl",
                "duration_s": 0.5,
            },
        )
        assert started.status_code == 200
        with client.websocket_connect("/ws/telemetry") as websocket:
            first = websocket.receive_json()
            second = websocket.receive_json()
        assert first["type"] == "telemetry"
        assert second["type"] == "telemetry"
        assert second["data"]["state"] in {"running", "completed"}
        assert first["data"]["telemetry"] is not None
        assert second["data"]["telemetry"] is not None
        assert second["data"]["telemetry"]["timestamp"] >= first["data"]["telemetry"]["timestamp"]


def test_phase7_completed_session_generates_summary() -> None:
    app = create_app()
    with TestClient(app) as client:
        started = client.post(
            "/api/session/start",
            json={
                "task": "point_to_point",
                "patient_profile": "severe",
                "mode": "fixed",
                "duration_s": 0.1,
            },
        )
        assert started.status_code == 200
        time.sleep(0.18)
        current = client.get("/api/session")
        assert current.status_code == 200
        assert current.json()["state"] == "completed"
        assert current.json()["report"]["completed"] is True
