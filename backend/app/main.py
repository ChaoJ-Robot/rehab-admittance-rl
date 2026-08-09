"""FastAPI/WebSocket entry point for the Phase 7 simulation page."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas.models import (
    AssignmentPayload,
    AssignmentRequest,
    ChatRequest,
    ModeRequest,
    PatientAssessmentPayload,
    PatientClinicalProfileRequest,
    PatientRequest,
    PatientSummary,
    SessionPrescriptionPayload,
    SessionSnapshot,
    StartRequest,
)
from backend.app.services.patient_store import validate_patient_id
from backend.app.services.session import TrainingSession


def _model_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def create_app() -> FastAPI:
    """Create an isolated application instance for deployment and tests."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        yield
        await application.state.session.aclose()

    app = FastAPI(title="Planar Rehab Training UI", version="0.1.0", lifespan=lifespan)
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

    @app.get("/api/patients")
    async def patients() -> list[PatientSummary]:
        """Return every registered patient's summary."""

        return app.state.session.list_patients()

    @app.get("/api/patients/{patient_id}")
    async def patient(patient_id: str) -> PatientSummary:
        """Return one patient's profile and training history."""

        if not validate_patient_id(patient_id):
            raise HTTPException(status_code=400, detail="invalid patient id")
        summary = app.state.session.patient_summary(patient_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return summary

    @app.post("/api/patients/{patient_id}")
    async def register_patient(patient_id: str, payload: PatientRequest) -> PatientSummary:
        """Create or update a patient profile before their first session."""

        if not validate_patient_id(patient_id):
            raise HTTPException(status_code=400, detail="invalid patient id")
        return app.state.session.register_patient(patient_id, payload.profile)

    @app.put("/api/patients/{patient_id}/clinical-profile")
    async def update_clinical_profile(
        patient_id: str, payload: PatientClinicalProfileRequest
    ) -> PatientSummary:
        """Replace the clinical context used for therapist decision support."""

        if not validate_patient_id(patient_id):
            raise HTTPException(status_code=400, detail="invalid patient id")
        summary = app.state.session.update_clinical_profile(
            patient_id, _model_dict(payload.clinical_profile)
        )
        if summary is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return summary

    @app.get("/api/patients/{patient_id}/assignments")
    async def assignments(patient_id: str) -> list[AssignmentPayload]:
        """Every training task dispatched to this patient, newest first."""

        if not validate_patient_id(patient_id):
            raise HTTPException(status_code=400, detail="invalid patient id")
        items = app.state.session.patient_assignments(patient_id)
        if items is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return items

    @app.post("/api/patients/{patient_id}/assignments")
    async def dispatch_assignment(patient_id: str, payload: AssignmentRequest) -> AssignmentPayload:
        """Therapist dispatches one training task to the patient."""

        if not validate_patient_id(patient_id):
            raise HTTPException(status_code=400, detail="invalid patient id")
        assignment = app.state.session.add_assignment(
            patient_id, payload.task, payload.task_params, payload.due_date
        )
        if assignment is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return assignment

    @app.get("/api/agent/assessment/{patient_id}")
    async def agent_assessment(patient_id: str) -> PatientAssessmentPayload:
        """Longitudinal trend assessment from the clinical agent layer."""

        if not validate_patient_id(patient_id):
            raise HTTPException(status_code=400, detail="invalid patient id")
        assessment = app.state.session.patient_assessment(patient_id)
        if assessment is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return assessment

    @app.get("/api/agent/prescription/{patient_id}")
    async def agent_prescription(patient_id: str) -> SessionPrescriptionPayload:
        """Agent suggestion (task/params/mode) for the patient's next session."""

        if not validate_patient_id(patient_id):
            raise HTTPException(status_code=400, detail="invalid patient id")
        prescription = app.state.session.patient_prescription(patient_id)
        if prescription is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return prescription

    @app.get("/api/session")
    async def session() -> SessionSnapshot:
        return app.state.session.snapshot()

    @app.post("/api/session/start")
    async def start(request: StartRequest) -> SessionSnapshot:
        try:
            return await app.state.session.start(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

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

    @app.get("/api/agent/events")
    async def agent_events() -> list[Any]:
        """Return all rule-based feedback events for audit/reporting."""

        return app.state.session.agent_events()

    @app.post("/api/agent/chat")
    async def agent_chat(request: ChatRequest) -> dict[str, str]:
        """Ask the LLM interaction agent a patient/therapist question."""

        reply = await app.state.session.chat(request.message)
        if reply is None:
            raise HTTPException(status_code=503, detail="LLM agent is not available")
        return {"message": reply}

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
