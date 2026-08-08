export type TaskName = "point_to_point" | "circle_tracking" | "figure8_tracking";
export type PatientProfile = "mild" | "moderate" | "severe";
export type ControlMode = "fixed" | "rl";
export type SessionState = "idle" | "running" | "paused" | "completed" | "stopped";

export interface Telemetry {
  timestamp: number;
  elapsed_s: number;
  task: TaskName;
  patient_profile: PatientProfile;
  mode: ControlMode;
  state: SessionState;
  reference_pose: number[];
  actual_pose: number[];
  tracking_error: number[];
  end_effector_velocity: number[];
  interaction_force: number[];
  human_power_w: number;
  fatigue: number;
  admittance_parameters: number[];
  rl_action: number[];
  task_progress: number;
  score: number;
  safety_status: "safe" | "fallback" | "paused" | "idle";
  safety_reasons: string[];
  agent_event: AgentEvent | null;
}

export interface AgentEvent {
  event: string;
  message: string;
  severity: "info" | "positive" | "warning" | "critical";
  timestamp_s: number;
  context: Record<string, unknown>;
}

export interface AgentSummary {
  title: string;
  message: string;
  highlights: string[];
  recommendation: string;
  event_count: number;
  source?: "rules" | "llm";
}

export interface AgentChatMessage {
  role: "user" | "agent";
  message: string;
  source: "rules" | "llm" | "user";
  timestamp_s: number;
}

export interface TrainingReport {
  completed: boolean;
  completion_rate: number;
  duration_s: number;
  average_tracking_error: number;
  peak_interaction_force: number;
  mean_interaction_force: number;
  motion_smoothness: number;
  patient_active_work: number;
  robot_assistance_work: number;
  parameter_change_total: number;
  safety_trigger_count: number;
  final_score: number;
}

export interface SessionSnapshot {
  session_id: string;
  state: SessionState;
  task: TaskName;
  patient_profile: PatientProfile;
  mode: ControlMode;
  elapsed_s: number;
  duration_s: number;
  task_progress: number;
  score: number;
  telemetry: Telemetry | null;
  report: TrainingReport | null;
  agent_event: AgentEvent | null;
  agent_summary: AgentSummary | null;
  agent_chat: AgentChatMessage[];
}

export interface ConfigSummary {
  tasks: TaskName[];
  patient_profiles: PatientProfile[];
  modes: ControlMode[];
  refresh_hz: number;
  simulation_only: boolean;
  hardware_validation_required: boolean;
  parameter_bounds: Record<string, number[]>;
  interaction_force_limit: number;
  task_speed_limit: number;
}
