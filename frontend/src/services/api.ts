import type {
  ConfigSummary,
  ControlMode,
  PatientAssessment,
  PatientClinicalProfile,
  PatientProfile,
  PatientSummary,
  SessionPrescription,
  SessionSnapshot,
  StartSessionParams,
  TaskAssignment,
  TaskName
} from "../types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const getConfig = () => request<ConfigSummary>("/api/config");
export const getSession = () => request<SessionSnapshot>("/api/session");

export function startSession(params: StartSessionParams) {
  return request<SessionSnapshot>("/api/session/start", {
    method: "POST",
    body: JSON.stringify(params)
  });
}

export const listPatients = () => request<PatientSummary[]>("/api/patients");

export function getPatient(patientId: string) {
  return request<PatientSummary>(`/api/patients/${encodeURIComponent(patientId)}`);
}

export function registerPatient(patientId: string, profile: PatientProfile) {
  return request<PatientSummary>(`/api/patients/${encodeURIComponent(patientId)}`, {
    method: "POST",
    body: JSON.stringify({ profile })
  });
}

export function updateClinicalProfile(patientId: string, clinicalProfile: PatientClinicalProfile) {
  return request<PatientSummary>(`/api/patients/${encodeURIComponent(patientId)}/clinical-profile`, {
    method: "PUT",
    body: JSON.stringify({ clinical_profile: clinicalProfile })
  });
}

export const getAssessment = (patientId: string) =>
  request<PatientAssessment>(`/api/agent/assessment/${encodeURIComponent(patientId)}`);

export const getPrescription = (patientId: string) =>
  request<SessionPrescription>(`/api/agent/prescription/${encodeURIComponent(patientId)}`);

export const getAssignments = (patientId: string) =>
  request<TaskAssignment[]>(`/api/patients/${encodeURIComponent(patientId)}/assignments`);

export function dispatchAssignment(
  patientId: string,
  payload: { task: TaskName; task_params: Record<string, number | string>; due_date: string }
) {
  return request<TaskAssignment>(`/api/patients/${encodeURIComponent(patientId)}/assignments`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export const pauseSession = () => request<SessionSnapshot>("/api/session/pause", { method: "POST" });
export const resumeSession = () => request<SessionSnapshot>("/api/session/resume", { method: "POST" });
export const stopSession = () => request<SessionSnapshot>("/api/session/stop", { method: "POST" });

export function setMode(mode: ControlMode) {
  return request<SessionSnapshot>("/api/session/mode", {
    method: "POST",
    body: JSON.stringify({ mode })
  });
}

export async function chatWithAgent(message: string): Promise<string> {
  const response = await request<{ message: string }>("/api/agent/chat", {
    method: "POST",
    body: JSON.stringify({ message })
  });
  return response.message;
}

export type { PatientProfile, TaskName };
