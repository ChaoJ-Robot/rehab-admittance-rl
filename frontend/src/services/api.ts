import type {
  ConfigSummary,
  ControlMode,
  PatientProfile,
  SessionSnapshot,
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

export function startSession(task: TaskName, patientProfile: PatientProfile, mode: ControlMode) {
  return request<SessionSnapshot>("/api/session/start", {
    method: "POST",
    body: JSON.stringify({ task, patient_profile: patientProfile, mode })
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
