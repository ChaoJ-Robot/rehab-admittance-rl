export type TaskName =
  | "point_to_point"
  | "circle_tracking"
  | "figure8_tracking"
  | "maze_navigation"
  | "color_memory"
  | "follow_to_reach"
  | "visual_guided_reach"
  | "motion_intercept"
  | "marker_memory";
export type PatientProfile = "mild" | "moderate" | "severe";
export type ControlMode = "fixed" | "rl";
export type SessionState = "idle" | "running" | "paused" | "completed" | "stopped";

export interface PatientClinicalProfile {
  diagnosis: string;
  affected_side: "left" | "right" | "bilateral" | "unknown";
  dominant_side: "left" | "right" | "unknown";
  onset_date: string;
  rehab_stage: "acute" | "subacute" | "chronic" | "unknown";
  goals: string[];
  precautions: string[];
  standardized_scores: Record<string, number>;
  notes: string;
}

export interface TrainingCheckIn {
  pain_vas: number;
  fatigue_0_10: number;
  exertion_rpe: number;
  note: string;
}

export interface TaskParamSpec {
  name: string;
  label: string;
  type: "slider" | "select";
  min?: number | null;
  max?: number | null;
  step?: number | null;
  default?: number | string | null;
  options?: Array<number | string>;
  labels?: Record<string, string>;
  unit?: string;
}

export interface StartSessionParams {
  task: TaskName;
  patient_id: string;
  patient_profile: PatientProfile;
  mode: ControlMode;
  duration_s?: number;
  task_params: Record<string, number | string>;
  assignment_id?: string;
  check_in: TrainingCheckIn;
}

export interface TaskAssignment {
  assignment_id: string;
  task: TaskName;
  task_params: Record<string, number | string>;
  due_date: string;
  status: "pending" | "completed";
  assigned_at: number;
  completed_at: number | null;
  completed_session: string | null;
}

export interface PatientHistoryEntry {
  session_id: string;
  task: TaskName;
  timestamp: number;
  duration_s: number;
  score: number;
  completion_rate: number;
  average_tracking_error: number;
  final_parameters: number[];
  mode: ControlMode;
  task_params: Record<string, number | string>;
  check_in: TrainingCheckIn;
  peak_interaction_force: number;
  active_participation_ratio: number;
  robot_assistance_ratio: number;
  safety_trigger_count: number;
  path_efficiency: number | null;
  collision_count: number;
  target_hit_count: number;
}

export interface PatientSummary {
  patient_id: string;
  profile: PatientProfile;
  created_at: number;
  last_session_at: number | null;
  session_count: number;
  latest_parameters: number[];
  history: PatientHistoryEntry[];
  clinical_profile: PatientClinicalProfile;
}

export type TrendClassification = "improving" | "plateau" | "regressing" | "insufficient_data";

export interface PatientAssessment {
  patient_id: string;
  sessions_analyzed: number;
  classification: TrendClassification;
  score_slope: number;
  completion_slope: number;
  error_slope: number;
  avg_score_recent: number | null;
  avg_completion_recent: number | null;
  avg_error_recent: number | null;
  flags: string[];
  narrative: string;
  risk_level: "low" | "moderate" | "high";
  evidence: string[];
}

export type DifficultyAction = "upgrade" | "maintain" | "downgrade" | "baseline";

export interface SessionPrescription {
  patient_id: string;
  task: TaskName;
  task_params: Record<string, number | string>;
  mode: ControlMode;
  difficulty_action: DifficultyAction;
  rationale: string[];
  risk_level: "low" | "moderate" | "high";
  confidence: number;
  missing_data: string[];
  precautions: string[];
  requires_doctor_approval: boolean;
}

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
  active_participation_ratio: number;
  robot_assistance_ratio: number;
  admittance_parameters: number[];
  rl_action: number[];
  task_progress: number;
  score: number;
  safety_status: "safe" | "fallback" | "paused" | "idle";
  safety_reasons: string[];
  agent_event: AgentEvent | null;
  maze_walls: number[][];
  maze_start: number[];
  maze_goal: number[];
  maze_optimal_path: number[][];
  maze_collision_count: number;
  maze_path_efficiency: number | null;
  task_success: boolean;
  memory_marker: number[];
  color_block_positions: number[][];
  color_block_names: string[];
  color_sequence: string[];
  task_phase: "memorize" | "recall" | null;
  task_targets: number[][];
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
  active_participation_ratio: number;
  robot_assistance_ratio: number;
  path_efficiency: number | null;
  collision_count: number;
  target_hit_count: number;
}

export interface SessionSnapshot {
  session_id: string;
  state: SessionState;
  task: TaskName;
  patient_id: string;
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
  task_params: Record<string, TaskParamSpec[]>;
  patient_profiles: PatientProfile[];
  modes: ControlMode[];
  refresh_hz: number;
  simulation_only: boolean;
  hardware_validation_required: boolean;
  parameter_bounds: Record<string, number[]>;
  interaction_force_limit: number;
  task_speed_limit: number;
}
