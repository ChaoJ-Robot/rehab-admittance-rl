from __future__ import annotations

import time

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from backend.app.main import create_app  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from rehab_sim.agent import LLMAgent, LLMClient, LLMConfig  # noqa: E402


def _llm_config(*, enabled: bool) -> LLMConfig:
    return LLMConfig(
        enabled=enabled,
        mode="hybrid",
        provider="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        timeout_s=5.0,
        event_enrichment_enabled=True,
        event_enrichment_cooldown_s=0.0,
        summary_enabled=True,
        chat_enabled=True,
        max_history_messages=10,
    )


def _llm_disabled_app() -> FastAPI:
    """App whose LLM layer is off so tests never hit the external API."""

    app = create_app()
    app.state.session._llm_agent = LLMAgent(_llm_config(enabled=False))
    return app


def _llm_stub_app(content: str) -> FastAPI:
    """App whose LLM client replies with fixed content through a mock transport."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    app = create_app()
    client = LLMClient(
        _llm_config(enabled=True), api_key="stub", transport=httpx.MockTransport(handler)
    )
    app.state.session._llm_agent = LLMAgent(_llm_config(enabled=True), client=client)
    return app


def test_phase7_rest_controls_and_report() -> None:
    app = _llm_disabled_app()
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
                "duration_s": 2.0,
            },
        )
        assert started.status_code == 200
        assert started.json()["state"] == "running"
        assert started.json()["agent_event"]["event"] == "task_started"

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
        events = client.get("/api/agent/events")
        assert events.status_code == 200
        assert events.json()[0]["event"] == "task_started"


def test_phase7_websocket_streams_telemetry() -> None:
    app = _llm_disabled_app()
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
        assert first["data"]["agent_event"] is not None


def test_phase7_completed_session_generates_summary() -> None:
    app = _llm_disabled_app()
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
        assert current.json()["agent_summary"]["title"] == "训练总结"


def test_agent_failure_does_not_stop_telemetry_or_control() -> None:
    app = _llm_disabled_app()

    def broken_observe(*_: object, **__: object) -> None:
        raise RuntimeError("agent unavailable")

    app.state.session._agent.observe = broken_observe  # type: ignore[method-assign]
    with TestClient(app) as client:
        started = client.post(
            "/api/session/start",
            json={
                "task": "point_to_point",
                "patient_profile": "moderate",
                "mode": "fixed",
                "duration_s": 2.0,
            },
        )
        assert started.status_code == 200
        time.sleep(0.08)
        current = client.get("/api/session").json()
        assert current["telemetry"] is not None
        assert current["telemetry"]["safety_status"] == "safe"
        client.post("/api/session/stop")


def test_llm_disabled_chat_returns_503_and_chat_feed_still_tracks_rules() -> None:
    app = _llm_disabled_app()
    with TestClient(app) as client:
        reply = client.post("/api/agent/chat", json={"message": "今天练得怎么样？"})
        assert reply.status_code == 503
        started = client.post(
            "/api/session/start",
            json={
                "task": "point_to_point",
                "patient_profile": "moderate",
                "mode": "fixed",
                "duration_s": 0.1,
            },
        )
        assert started.status_code == 200
        time.sleep(0.18)
        current = client.get("/api/session").json()
        assert current["state"] == "completed"
        assert len(current["agent_chat"]) >= 2
        assert current["agent_chat"][0]["role"] == "agent"
        assert current["agent_chat"][0]["source"] == "rules"
        assert current["agent_summary"]["source"] == "rules"


def test_llm_chat_and_summary_with_stub_client() -> None:
    app = _llm_stub_app(
        '{"title": "训练总结", "message": "表现良好，继续保持。", '
        '"highlights": ["误差低"], "recommendation": "下次可略微增加难度"}'
    )
    with TestClient(app) as client:
        reply = client.post("/api/agent/chat", json={"message": "今天练得怎么样？"})
        assert reply.status_code == 200
        assert reply.json()["message"] == "表现良好，继续保持。"
        started = client.post(
            "/api/session/start",
            json={
                "task": "point_to_point",
                "patient_profile": "moderate",
                "mode": "fixed",
                "duration_s": 0.1,
            },
        )
        assert started.status_code == 200
        time.sleep(0.5)
        current = client.get("/api/session").json()
        assert current["state"] == "completed"
        assert current["agent_summary"]["source"] == "llm"
        sources = {item["source"] for item in current["agent_chat"]}
        assert "llm" in sources and "rules" in sources
