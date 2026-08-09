"""Persistent per-patient training records.

Each patient owns one JSON file under ``data/patients/``. The file stores the
patient's virtual profile, the latest admittance parameters (so the next
session resumes from where the previous one stopped instead of starting from
the generic baseline) and a short session history for the therapist page.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("rehab.backend.patients")

PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_FILE_SUFFIX = ".json"


def validate_patient_id(patient_id: str) -> bool:
    """Return whether ``patient_id`` is safe to use as a file name."""

    return bool(PATIENT_ID_PATTERN.fullmatch(patient_id))


@dataclass
class PatientHistoryEntry:
    """One completed training session appended to a patient record."""

    session_id: str
    task: str
    timestamp: float
    duration_s: float
    score: float
    completion_rate: float
    average_tracking_error: float
    final_parameters: list[float] = field(default_factory=list)
    mode: str = "fixed"
    task_params: dict[str, Any] = field(default_factory=dict)
    check_in: dict[str, Any] = field(default_factory=dict)
    peak_interaction_force: float = 0.0
    active_participation_ratio: float = 0.0
    robot_assistance_ratio: float = 0.0
    safety_trigger_count: int = 0
    path_efficiency: float | None = None
    collision_count: int = 0
    target_hit_count: int = 0


@dataclass
class PatientClinicalProfile:
    """Disease-specific context kept separate from the simulation severity profile."""

    diagnosis: str = ""
    affected_side: str = "unknown"
    dominant_side: str = "unknown"
    onset_date: str = ""
    rehab_stage: str = "unknown"
    goals: list[str] = field(default_factory=list)
    precautions: list[str] = field(default_factory=list)
    standardized_scores: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> PatientClinicalProfile:
        raw = mapping if isinstance(mapping, Mapping) else {}
        scores = raw.get("standardized_scores", {})
        return cls(
            diagnosis=str(raw.get("diagnosis", "")),
            affected_side=str(raw.get("affected_side", "unknown")),
            dominant_side=str(raw.get("dominant_side", "unknown")),
            onset_date=str(raw.get("onset_date", "")),
            rehab_stage=str(raw.get("rehab_stage", "unknown")),
            goals=[str(value) for value in raw.get("goals", [])],
            precautions=[str(value) for value in raw.get("precautions", [])],
            standardized_scores=(
                {str(key): float(value) for key, value in scores.items()}
                if isinstance(scores, Mapping)
                else {}
            ),
            notes=str(raw.get("notes", "")),
        )


@dataclass
class TaskAssignment:
    """One training task dispatched to the patient by the therapist."""

    assignment_id: str
    task: str
    task_params: dict[str, Any] = field(default_factory=dict)
    due_date: str = ""
    status: str = "pending"
    assigned_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    completed_session: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> TaskAssignment:
        params = mapping.get("task_params", {})
        return cls(
            assignment_id=str(mapping["assignment_id"]),
            task=str(mapping["task"]),
            task_params=dict(params) if isinstance(params, Mapping) else {},
            due_date=str(mapping.get("due_date", "")),
            status=str(mapping.get("status", "pending")),
            assigned_at=float(mapping.get("assigned_at", time.time())),
            completed_at=(float(mapping["completed_at"]) if mapping.get("completed_at") else None),
            completed_session=(
                str(mapping["completed_session"]) if mapping.get("completed_session") else None
            ),
        )


@dataclass
class PatientRecord:
    """A patient's persistent profile and cross-session state."""

    patient_id: str
    profile: str
    created_at: float
    latest_parameters: list[float] = field(default_factory=list)
    last_session_at: float | None = None
    session_count: int = 0
    history: list[PatientHistoryEntry] = field(default_factory=list)
    assignments: list[TaskAssignment] = field(default_factory=list)
    clinical_profile: PatientClinicalProfile = field(default_factory=PatientClinicalProfile)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> PatientRecord:
        history = [
            PatientHistoryEntry(**entry)
            for entry in mapping.get("history", [])
            if isinstance(entry, dict)
        ]
        assignments = [
            TaskAssignment.from_mapping(entry)
            for entry in mapping.get("assignments", [])
            if isinstance(entry, dict)
        ]
        return cls(
            patient_id=str(mapping["patient_id"]),
            profile=str(mapping["profile"]),
            created_at=float(mapping["created_at"]),
            latest_parameters=[float(value) for value in mapping.get("latest_parameters", [])],
            last_session_at=(
                float(mapping["last_session_at"]) if mapping.get("last_session_at") else None
            ),
            session_count=int(mapping.get("session_count", 0)),
            history=history,
            assignments=assignments,
            clinical_profile=PatientClinicalProfile.from_mapping(mapping.get("clinical_profile")),
        )


class PatientStore:
    """Load and save patient records as JSON files on disk."""

    def __init__(self, root: Path) -> None:
        self._directory = Path(root) / "data" / "patients"
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path(self, patient_id: str) -> Path:
        if not validate_patient_id(patient_id):
            raise ValueError(f"invalid patient id: {patient_id!r}")
        return self._directory / f"{patient_id}{_FILE_SUFFIX}"

    def load(self, patient_id: str) -> PatientRecord | None:
        """Return the patient record, or None when it does not exist yet."""

        path = self._path(patient_id)
        if not path.is_file():
            return None
        try:
            mapping = json.loads(path.read_text(encoding="utf-8"))
            return PatientRecord.from_mapping(mapping)
        except (OSError, KeyError, TypeError, ValueError) as error:
            LOGGER.warning("patient_record_unreadable id=%s error=%s", patient_id, error)
            return None

    def save(self, record: PatientRecord) -> None:
        """Atomically persist one patient record."""

        path = self._path(record.patient_id)
        payload = json.dumps(asdict(record), ensure_ascii=False, indent=2)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    def create(self, patient_id: str, profile: str) -> PatientRecord:
        """Create a fresh record for a brand-new patient."""

        return PatientRecord(
            patient_id=patient_id,
            profile=profile,
            created_at=time.time(),
        )

    def list_ids(self) -> list[str]:
        """Return the ids of all registered patients, sorted by name."""

        return sorted(
            path.stem
            for path in self._directory.glob(f"*{_FILE_SUFFIX}")
            if validate_patient_id(path.stem)
        )
