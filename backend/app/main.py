"""FastAPI/WebSocket entry point for the Phase 7 simulation page."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas.models import ModeRequest, SessionSnapshot, StartRequest
from backend.app.services.session import TrainingSession


def _model_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def create_app() -> FastAPI:
    """Create an isolated application instance for deployment and tests."""

    app = FastAPI(title="Planar Rehab Training UI", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.session = TrainingSession()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": "simulation_only"}

    @app.get("/api/config")
    async def config() -> Any:
        return app.state.session.config_summary()

    @app.get("/api/session")
    async def session() -> SessionSnapshot:
        return app.state.session.snapshot()

    @app.post("/api/session/start")
    async def start(request: StartRequest) -> SessionSnapshot:
        return await app.state.session.start(request)

    @app.post("/api/session/pause")
    async def pause() -> SessionSnapshot:
        try:
            return app.state.session.pause()
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/session/resume")
    async def resume() -> SessionSnapshot:
        try:
            return app.state.session.resume()
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/session/stop")
    async def stop() -> SessionSnapshot:
        return app.state.session.stop()

    @app.post("/api/session/mode")
    async def mode(request: ModeRequest) -> SessionSnapshot:
        return app.state.session.set_mode(request.mode)

    @app.get("/api/report")
    async def report() -> Any:
        snapshot = app.state.session.snapshot()
        if snapshot.report is None:
            raise HTTPException(status_code=404, detail="training report is not available")
        return snapshot.report

    @app.websocket("/ws/telemetry")
    async def telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                snapshot = app.state.session.snapshot()
                await websocket.send_json({"type": "telemetry", "data": _model_dict(snapshot)})
                await asyncio.sleep(1.0 / TrainingSession.refresh_hz)
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            raise

    return app


app = create_app()
