import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { LineChart } from "./components/LineChart";
import { TrajectoryChart } from "./components/TrajectoryChart";
import { MazeChart } from "./components/MazeChart";
import { ColorMemoryPanel } from "./components/ColorMemoryPanel";
import {
  chatWithAgent,
  dispatchAssignment,
  getAssessment,
  getConfig,
  getPatient,
  getPrescription,
  getSession,
  getAssignments,
  listPatients,
  pauseSession,
  registerPatient,
  resumeSession,
  setMode,
  startSession,
  stopSession,
  updateClinicalProfile
} from "./services/api";
import type {
  AgentChatMessage,
  ConfigSummary,
  ControlMode,
  DifficultyAction,
  PatientAssessment,
  PatientClinicalProfile,
  PatientProfile,
  PatientSummary,
  SessionPrescription,
  SessionSnapshot,
  TaskAssignment,
  TaskName,
  TaskParamSpec,
  Telemetry,
  TrainingCheckIn,
  TrendClassification
} from "./types";
import "./styles.css";

const taskLabels: Record<TaskName, string> = {
  point_to_point: "点到点训练",
  circle_tracking: "圆轨迹训练",
  figure8_tracking: "八字轨迹训练",
  maze_navigation: "迷宫导航",
  color_memory: "色块记忆",
  follow_to_reach: "跟随到达",
  visual_guided_reach: "视觉引导到达",
  motion_intercept: "运动拦截",
  marker_memory: "目标标记记忆"
};

const taskMeta: Record<TaskName, { category: string; desc: string; level: string }> = {
  point_to_point: { category: "轨迹跟踪", desc: "从起点平稳移动至目标点，训练运动启动与定位控制能力。", level: "基础" },
  circle_tracking: { category: "轨迹跟踪", desc: "沿圆形参考轨迹连续运动，训练节律性与速度适应能力。", level: "进阶" },
  figure8_tracking: { category: "轨迹跟踪", desc: "沿八字轨迹换向跟踪，训练方向切换与协调控制。", level: "进阶" },
  maze_navigation: { category: "空间导航", desc: "引导末端穿越迷宫走廊，支持多种地图，训练空间规划能力。", level: "挑战" },
  color_memory: { category: "认知训练", desc: "记忆颜色序列并按顺序复述，记忆与运动的双任务训练。", level: "挑战" },
  follow_to_reach: { category: "目标到达", desc: "按顺序跟随并到达多个目标点，训练运动规划与定位控制。", level: "进阶" },
  visual_guided_reach: { category: "目标到达", desc: "仅显示当前目标点，按视觉引导逐个到达，训练手眼协调与定位精度。", level: "进阶" },
  motion_intercept: { category: "动态跟踪", desc: "跟踪拦截沿直线走廊往复移动的目标，训练动态跟踪与时机判断。", level: "挑战" },
  marker_memory: { category: "认知训练", desc: "记忆标记位置，消失后移动到记忆位置，空间记忆与运动双任务。", level: "挑战" }
};

const patientLabels: Record<PatientProfile, string> = {
  mild: "轻度患者",
  moderate: "中度患者",
  severe: "重度患者"
};

const stateLabels: Record<SessionSnapshot["state"], string> = {
  idle: "待机",
  running: "训练中",
  paused: "已暂停",
  completed: "已完成",
  stopped: "已停止"
};

const classificationLabels: Record<TrendClassification, string> = {
  improving: "持续改善",
  plateau: "平台期",
  regressing: "表现下滑",
  insufficient_data: "数据不足"
};

const actionLabels: Record<DifficultyAction, string> = {
  upgrade: "加量",
  maintain: "维持",
  downgrade: "降量",
  baseline: "基线"
};

const emptyClinicalProfile: PatientClinicalProfile = {
  diagnosis: "",
  affected_side: "unknown",
  dominant_side: "unknown",
  onset_date: "",
  rehab_stage: "unknown",
  goals: [],
  precautions: [],
  standardized_scores: {},
  notes: ""
};

const initialCheckIn: TrainingCheckIn = {
  pain_vas: 0,
  fatigue_0_10: 0,
  exertion_rpe: 0,
  note: ""
};

const initialSnapshot: SessionSnapshot = {
  session_id: "none",
  state: "idle",
  task: "point_to_point",
  patient_id: "default",
  patient_profile: "moderate",
  mode: "fixed",
  elapsed_s: 0,
  duration_s: 4,
  task_progress: 0,
  score: 0,
  telemetry: null,
  report: null,
  agent_event: null,
  agent_summary: null,
  agent_chat: []
};

function App() {
  const [config, setConfig] = useState<ConfigSummary | null>(null);
  const [snapshot, setSnapshot] = useState<SessionSnapshot>(initialSnapshot);
  const [history, setHistory] = useState<Telemetry[]>([]);
  const [view, setView] = useState<"home" | "training" | "patient">("home");
  const [originView, setOriginView] = useState<"home" | "patient">("home");
  const [task, setTask] = useState<TaskName>("point_to_point");
  const [patient, setPatient] = useState<PatientProfile>("moderate");
  const [patientId, setPatientId] = useState("");
  const [loadedPatient, setLoadedPatient] = useState<PatientSummary | null>(null);
  const [assessment, setAssessment] = useState<PatientAssessment | null>(null);
  const [prescription, setPrescription] = useState<SessionPrescription | null>(null);
  const [assignments, setAssignments] = useState<TaskAssignment[]>([]);
  const [dueDate, setDueDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [dispatched, setDispatched] = useState<string | null>(null);
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [patientLoading, setPatientLoading] = useState(false);
  const [taskParams, setTaskParams] = useState<Record<string, number | string>>({});
  const [mode, setModeValue] = useState<ControlMode>("fixed");
  const [error, setError] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [chatSending, setChatSending] = useState(false);
  const [clinicalDraft, setClinicalDraft] = useState<PatientClinicalProfile>(emptyClinicalProfile);
  const [clinicalEditing, setClinicalEditing] = useState(false);
  const [clinicalSaving, setClinicalSaving] = useState(false);
  const [checkIn, setCheckIn] = useState<TrainingCheckIn>(initialCheckIn);
  const [taskCategory, setTaskCategory] = useState("全部");
  const chatFeedRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void Promise.all([getConfig(), getSession()])
      .then(([loadedConfig, loadedSession]) => {
        setConfig(loadedConfig);
        setSnapshot(loadedSession);
        setModeValue(loadedSession.mode);
        setPatientId(loadedSession.patient_id === "default" ? "" : loadedSession.patient_id);
        if (loadedSession.state === "running" || loadedSession.state === "paused") {
          setView("training");
        }
      })
      .catch((reason: unknown) => setError(String(reason)));

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws/telemetry`);
    let socketActive = true;
    socket.onopen = () => {
      if (!socketActive) return;
      setError((previous) => (previous?.includes("WebSocket") ? null : previous));
    };
    socket.onmessage = (event) => {
      if (!socketActive) return;
      const message = JSON.parse(event.data) as { type: string; data: SessionSnapshot };
      if (message.type !== "telemetry") return;
      setError((previous) => (previous && previous.includes("WebSocket") ? null : previous));
      setSnapshot(message.data);
      if (message.data.telemetry) {
        setHistory((previous) => [...previous, message.data.telemetry as Telemetry].slice(-240));
      }
    };
    socket.onerror = () => {
      if (socketActive) setError("WebSocket 连接失败，请确认 FastAPI 后端已启动");
    };
    return () => {
      socketActive = false;
      socket.close();
    };
  }, []);

  useEffect(() => {
    if (view === "training") return;
    let cancelled = false;
    if (view === "home") {
      void listPatients()
        .then((items) => { if (!cancelled) setPatients(items); })
        .catch(() => undefined);
    }
    const loadedId = loadedPatient?.patient_id ?? "";
    if (loadedId) {
      void getAssignments(loadedId)
        .then((items) => { if (!cancelled) setAssignments(items); })
        .catch(() => undefined);
    } else if (!cancelled) {
      setAssignments([]);
    }
    return () => { cancelled = true; };
  }, [view, loadedPatient]);

  useEffect(() => {
    if (!config) return;
    const specs = config.task_params?.[task] ?? [];
    setTaskParams((previous) => {
      const defaults: Record<string, number | string> = {};
      for (const spec of specs) {
        if (spec.default !== null && spec.default !== undefined) {
          defaults[spec.name] = spec.default;
        }
      }
      return { ...defaults, ...previous };
    });
  }, [config, task]);

  const runRequest = async (operation: () => Promise<SessionSnapshot>) => {
    try {
      setError(null);
      const next = await operation();
      setSnapshot(next);
      if (next.state === "running" && next.elapsed_s === 0) setHistory([]);
      return next;
    } catch (reason) {
      setError(String(reason));
      return null;
    }
  };

  const handleTaskSelect = (name: TaskName) => {
    setTask(name);
    const specs = config?.task_params?.[name] ?? [];
    const defaults: Record<string, number | string> = {};
    for (const spec of specs) {
      if (spec.default !== null && spec.default !== undefined) {
        defaults[spec.name] = spec.default;
      }
    }
    setTaskParams(defaults);
  };

  const loadPatientById = async (id: string) => {
    setPatientLoading(true);
    setError(null);
    try {
      const summary = await getPatient(id);
      setLoadedPatient(summary);
      setPatient(summary.profile);
      setClinicalDraft(summary.clinical_profile ?? emptyClinicalProfile);
      const [nextAssessment, nextPrescription, nextAssignments] = await Promise.all([
        getAssessment(id).catch(() => null),
        getPrescription(id).catch(() => null),
        getAssignments(id).catch(() => [] as TaskAssignment[])
      ]);
      setAssessment(nextAssessment);
      setPrescription(nextPrescription);
      setAssignments(nextAssignments);
      setDispatched(null);
    } catch (reason) {
      void reason;
      setLoadedPatient(null);
      setAssessment(null);
      setPrescription(null);
      setClinicalDraft(emptyClinicalProfile);
      setError(`未找到患者 ${id}，将作为新患者建档`);
    } finally {
      setPatientLoading(false);
    }
  };

  const handleLoadPatient = () => {
    const id = patientId.trim();
    if (!id) return;
    void loadPatientById(id);
  };

  const adoptPrescription = () => {
    if (!prescription) return;
    setTask(prescription.task);
    const specs = config?.task_params?.[prescription.task] ?? [];
    const defaults: Record<string, number | string> = {};
    for (const spec of specs) {
      if (spec.default !== null && spec.default !== undefined) {
        defaults[spec.name] = spec.default;
      }
    }
    setTaskParams({ ...defaults, ...prescription.task_params });
    if (prescription.mode !== mode) {
      setModeValue(prescription.mode);
      void runRequest(() => setMode(prescription.mode));
    }
  };

  const handleStart = async (origin: "home" | "patient", assignmentId?: string) => {
    const id = patientId.trim();
    if (!id) {
      setError("请先输入患者 ID 再开始训练");
      return;
    }
    if (!loadedPatient || loadedPatient.patient_id !== id) {
      try {
        setLoadedPatient(await registerPatient(id, patient));
      } catch (reason) {
        void reason;
      }
    }
    const next = await runRequest(() =>
      startSession({ task, patient_id: id, patient_profile: patient, mode, task_params: taskParams, assignment_id: assignmentId, check_in: checkIn })
    );
    if (next) {
      setOriginView(origin);
      setView("training");
    }
  };

  const saveClinicalProfile = async () => {
    if (!loadedPatient || clinicalSaving) return;
    setClinicalSaving(true);
    setError(null);
    try {
      const updated = await updateClinicalProfile(loadedPatient.patient_id, clinicalDraft);
      setLoadedPatient(updated);
      setClinicalDraft(updated.clinical_profile);
      setClinicalEditing(false);
      const [nextAssessment, nextPrescription] = await Promise.all([
        getAssessment(updated.patient_id),
        getPrescription(updated.patient_id)
      ]);
      setAssessment(nextAssessment);
      setPrescription(nextPrescription);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setClinicalSaving(false);
    }
  };

  const handleDispatch = async () => {
    if (!loadedPatient) return;
    setError(null);
    try {
      await dispatchAssignment(loadedPatient.patient_id, { task, task_params: taskParams, due_date: dueDate });
      setAssignments(await getAssignments(loadedPatient.patient_id));
      setDispatched(`已向 ${loadedPatient.patient_id} 派发“${taskLabels[task]}”，截止日期 ${dueDate || "未设置"}。`);
    } catch (reason) {
      setError(String(reason));
    }
  };

  const enterAssignment = async (item: TaskAssignment) => {
    const id = patientId.trim();
    if (!id) {
      setError("请先输入患者 ID 再进入任务");
      return;
    }
    const specs = config?.task_params?.[item.task] ?? [];
    const defaults: Record<string, number | string> = {};
    for (const spec of specs) {
      if (spec.default !== null && spec.default !== undefined) {
        defaults[spec.name] = spec.default;
      }
    }
    const merged = { ...defaults, ...item.task_params };
    setTask(item.task);
    setTaskParams(merged);
    const profile = loadedPatient?.profile ?? patient;
    const next = await runRequest(() =>
      startSession({ task: item.task, patient_id: id, patient_profile: profile, mode, task_params: merged, assignment_id: item.assignment_id, check_in: checkIn })
    );
    if (next) {
      setOriginView("patient");
      setView("training");
    }
  };

  const sendChat = async () => {
    const text = chatInput.trim();
    if (!text || chatSending) return;
    setChatSending(true);
    setError(null);
    try {
      await chatWithAgent(text);
      setSnapshot(await getSession());
    } catch (reason) {
      setError(String(reason));
    } finally {
      setChatSending(false);
      setChatInput("");
    }
  };

  const chatFeed = (snapshot.agent_chat ?? [])
    .filter((item, index, items) => index === 0 || item.message !== items[index - 1].message)
    .slice(-30);

  useEffect(() => {
    const feed = chatFeedRef.current;
    if (feed) feed.scrollTop = feed.scrollHeight;
  }, [chatFeed.length]);

  const telemetry = snapshot.telemetry;
  const agentEvent = snapshot.agent_event ?? telemetry?.agent_event ?? null;
  const running = snapshot.state === "running";
  const paused = snapshot.state === "paused";
  const finished = snapshot.state === "completed" || snapshot.state === "stopped";
  const safetyStatus = telemetry?.safety_status ?? "idle";
  const safetyLabel =
    safetyStatus === "fallback" ? "保护回退" : safetyStatus === "safe" ? "系统安全" : "等待遥测";
  const connectionLabel = error ? "连接异常" : telemetry ? "实时连接" : "等待连接";
  const forceMagnitude = Math.hypot(telemetry?.interaction_force[0] ?? 0, telemetry?.interaction_force[1] ?? 0);
  const modeLabel = snapshot.mode === "rl" ? "RL 参数调节" : "固定导纳";
  const progressPercent = Math.round(snapshot.task_progress * 100);
  const forceSeries = useMemo(
    () => [0, 1, 2].map((axis) => history.map((point) => point.interaction_force[axis])),
    [history]
  );
  const parameterSeries = useMemo(
    () => [0, 1, 2, 3, 4].map((axis) => history.map((point) => point.admittance_parameters[axis])),
    [history]
  );
  const lastFrame = history.length > 0 ? history[history.length - 1] : telemetry;
  const chartTargets =
    snapshot.task === "marker_memory"
      ? lastFrame?.memory_marker?.length
        ? [lastFrame.memory_marker]
        : []
      : snapshot.task === "follow_to_reach" ||
          snapshot.task === "visual_guided_reach" ||
          snapshot.task === "motion_intercept"
        ? lastFrame?.task_targets ?? []
        : [];

  const topbar = (
    <header className="topbar">
      <div className="brand-lockup">
        <img src="/buaa-logo.png" className="brand-logo" alt="北京航空航天大学" />
        <div className="brand-title">
          <h1>上肢康复训练<span>控制台</span></h1>
          <small>PLANAR REHAB · ADAPTIVE ADMITTANCE</small>
        </div>
      </div>
      <div className="topbar-actions">
        {view !== "training" && (
          <div className="role-tabs">
            <button className={`role-tab ${view === "home" ? "active" : ""}`} onClick={() => setView("home")}>医生终端</button>
            <button className={`role-tab ${view === "patient" ? "active" : ""}`} onClick={() => setView("patient")}>患者终端</button>
          </div>
        )}
        <div className="connection-chip">
          <span className={`status-dot ${telemetry ? safetyStatus : "idle"}`} />
          <span>{connectionLabel}</span>
          <small>20 Hz</small>
        </div>
        <div className="simulation-tag"><span className="shield-mark">◆</span> SIMULATION ONLY</div>
      </div>
    </header>
  );

  if (view === "home") {
    const tasks = config?.tasks ?? (Object.keys(taskLabels) as TaskName[]);
    const taskCategories = ["全部", ...Array.from(new Set(tasks.map((name) => taskMeta[name].category)))];
    const visibleTasks = taskCategory === "全部"
      ? tasks
      : tasks.filter((name) => taskMeta[name].category === taskCategory);
    const taskSpecs = config?.task_params?.[task] ?? [];
    const totalSessions = patients.reduce((sum, item) => sum + item.session_count, 0);
    const lastSessionAt = patients.reduce<number | null>(
      (latest, item) =>
        item.last_session_at && item.last_session_at > (latest ?? 0) ? item.last_session_at : latest,
      null
    );
    const boardPatients = [...patients].sort(
      (a, b) => (b.last_session_at ?? b.created_at) - (a.last_session_at ?? a.created_at)
    );
    const recentAvgScore = loadedPatient ? averageScore(loadedPatient.history, 3) : null;
    const recentHistory = loadedPatient ? [...loadedPatient.history].reverse().slice(0, 5) : [];
    return (
      <main className="app-shell">
        {topbar}
        {error && <div className="error-banner"><span className="alert-icon">!</span><span>{error}</span></div>}

        <div className="home-layout">
          <section className="home-hero">
            <div className="hero-copy">
              <h2>为患者选择训练方案<em>，开始本次康复。</em></h2>
              <p>管理患者档案、配置训练任务与参数、实时监控训练过程；安全强化学习调节导纳参数，智能教练全程辅助。</p>
            </div>
            <div className="hero-stats">
              <div className="stat-mini"><span>在档患者</span><strong>{patients.length} 位</strong></div>
              <div className="stat-mini"><span>累计训练</span><strong>{totalSessions} 次</strong></div>
              <div className="stat-mini"><span>最近训练</span><strong>{formatWhen(lastSessionAt)}</strong></div>
              <div className="stat-mini"><span>遥测状态</span><strong className={telemetry ? "stat-ok" : "stat-idle"}>{telemetry ? "20 Hz 实时" : "等待连接"}</strong></div>
            </div>
          </section>

          <div className="home-grid">
            <div className="home-side">
              <section className="patient-panel card">
                <div className="patient-panel-head">
                  <h3>患者档案</h3>
                  <small>PATIENT DATABASE</small>
                </div>
                <div className="patient-id-row">
                  <input
                    value={patientId}
                    onChange={(event) => setPatientId(event.target.value)}
                    onKeyDown={(event) => { if (event.key === "Enter") handleLoadPatient(); }}
                    placeholder="输入患者 ID，如 P001"
                    maxLength={32}
                  />
                  <button className="secondary action-button" onClick={handleLoadPatient} disabled={patientLoading}>
                    {patientLoading ? "加载中…" : "加载档案"}
                  </button>
                </div>
                {loadedPatient ? (
                  <>
                    <div className="patient-summary">
                      <div className="patient-summary-item"><span>档案</span><strong>{loadedPatient.patient_id} · {patientLabels[loadedPatient.profile]}</strong></div>
                      <div className="patient-summary-item"><span>累计训练</span><strong>{loadedPatient.session_count} 次</strong></div>
                      <div className="patient-summary-item"><span>上次训练</span><strong>{loadedPatient.last_session_at ? new Date(loadedPatient.last_session_at * 1000).toLocaleString("zh-CN") : "尚无记录"}</strong></div>
                      <div className="patient-summary-item"><span>近 3 次平均得分</span><strong>{recentAvgScore !== null ? recentAvgScore.toFixed(2) : "尚无记录"}</strong></div>
                      <div className="patient-summary-item"><span>上次导纳参数</span><strong>{loadedPatient.latest_parameters.length ? loadedPatient.latest_parameters.map((value) => value.toFixed(2)).join(" / ") : "默认基线（首次训练）"}</strong></div>
                    </div>
                    {recentHistory.length > 0 && (
                      <div className="patient-history">
                        <div className="history-title">历史训练记录</div>
                        <table>
                          <thead>
                            <tr><th>日期</th><th>任务</th><th>时长</th><th>得分</th><th>完成率</th></tr>
                          </thead>
                          <tbody>
                            {recentHistory.map((entry) => (
                              <tr key={entry.session_id}>
                                <td>{formatWhen(entry.timestamp)}</td>
                                <td>{taskLabels[entry.task]}</td>
                                <td>{entry.duration_s.toFixed(0)}s</td>
                                <td>{entry.score.toFixed(1)}</td>
                                <td>{Math.round(entry.completion_rate * 100)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="patient-empty">尚未选择患者。从下方“患者总览”点选一位，或在上方输入新 ID 建档。</div>
                )}
              </section>

              {loadedPatient && (
                <ClinicalProfileCard
                  value={clinicalDraft}
                  editing={clinicalEditing}
                  saving={clinicalSaving}
                  onEdit={() => setClinicalEditing(true)}
                  onCancel={() => {
                    setClinicalDraft(loadedPatient.clinical_profile ?? emptyClinicalProfile);
                    setClinicalEditing(false);
                  }}
                  onChange={setClinicalDraft}
                  onSave={() => void saveClinicalProfile()}
                />
              )}

              {loadedPatient && (
                <section className="agent-panel card">
                  <div className="patient-panel-head">
                    <h3>Agent 临床评估</h3>
                    <small>CLINICAL ASSISTANT</small>
                  </div>
                  {assessment ? (
                    <div className="agent-assessment">
                      <div className="agent-classification">
                        <span className={`class-badge class-${assessment.classification}`}>
                          {classificationLabels[assessment.classification]}
                        </span>
                        <span className={`risk-badge risk-${assessment.risk_level}`}>
                          {assessment.risk_level === "high" ? "高风险" : assessment.risk_level === "moderate" ? "需关注" : "低风险"}
                        </span>
                        <span className="agent-narrative">{assessment.narrative}</span>
                      </div>
                      {assessment.sessions_analyzed >= 3 && (
                        <div className="agent-metrics">
                          <div className="board-metric">
                            <span>近 3 次完成率</span>
                            <strong>{assessment.avg_completion_recent !== null ? `${(assessment.avg_completion_recent * 100).toFixed(0)}%` : "—"}</strong>
                          </div>
                          <div className="board-metric">
                            <span>近 3 次均分</span>
                            <strong>{assessment.avg_score_recent !== null ? assessment.avg_score_recent.toFixed(2) : "—"}</strong>
                          </div>
                          <div className="board-metric">
                            <span>平均轨迹误差</span>
                            <strong>{assessment.avg_error_recent !== null ? assessment.avg_error_recent.toFixed(4) : "—"}</strong>
                          </div>
                        </div>
                      )}
                      {assessment.flags.length > 0 && (
                        <ul className="agent-flags">
                          {assessment.flags.map((flag) => <li key={flag}>{flag}</li>)}
                        </ul>
                      )}
                      {assessment.evidence.length > 0 && (
                        <div className="agent-evidence">依据：{assessment.evidence.join("；")}</div>
                      )}
                    </div>
                  ) : (
                    <div className="patient-empty">历史不足 3 次，完成更多训练后自动生成趋势评估。</div>
                  )}
                  {prescription && (
                    <div className="agent-prescription">
                      <div className="rx-title">
                        下次训练处方建议
                        <span className={`rx-action rx-${prescription.difficulty_action}`}>{actionLabels[prescription.difficulty_action]}</span>
                        <span className={`risk-badge risk-${prescription.risk_level}`}>
                          置信度 {Math.round(prescription.confidence * 100)}%
                        </span>
                      </div>
                      <div className="rx-rows">
                        <div className="rx-row"><span>任务</span><strong>{taskLabels[prescription.task]}</strong></div>
                        <div className="rx-row"><span>模式</span><strong>{prescription.mode === "rl" ? "RL 参数调节" : "固定导纳"}</strong></div>
                        {Object.entries(prescription.task_params).map(([name, value]) => {
                          const spec = (config?.task_params?.[prescription.task] ?? []).find((item) => item.name === name);
                          return (
                            <div className="rx-row" key={name}>
                              <span>{spec?.label ?? name}</span>
                              <strong>{value}{spec?.unit ? ` ${spec.unit}` : ""}</strong>
                            </div>
                          );
                        })}
                      </div>
                      <ul className="rx-rationale">
                        {prescription.rationale.map((reason) => <li key={reason}>{reason}</li>)}
                      </ul>
                      {prescription.missing_data.length > 0 && (
                        <div className="rx-missing">建议补充：{prescription.missing_data.join("、")}</div>
                      )}
                      {prescription.precautions.length > 0 && (
                        <div className="rx-precautions">注意事项：{prescription.precautions.join("；")}</div>
                      )}
                      <button className="secondary rx-adopt" onClick={adoptPrescription}>
                        采纳并填入训练设置
                      </button>
                      <small className="rx-note">处方由 Agent 基于历史数据自动生成，最终决策由医生确认。</small>
                    </div>
                  )}
                </section>
              )}

              <section className="home-controls card">
                <div className="patient-panel-head">
                  <h3>训练设置</h3>
                  <small>SESSION SETUP</small>
                </div>
                <div className="home-control-fields">
                  <div className="control-field">
                    <label>患者配置</label>
                    <select value={patient} onChange={(event) => setPatient(event.target.value as PatientProfile)}>
                      {(config?.patient_profiles ?? ["moderate"]).map((item) => (
                        <option key={item} value={item}>{patientLabels[item]}</option>
                      ))}
                    </select>
                  </div>
                  <div className="control-field">
                    <label>控制模式</label>
                    <select value={mode} onChange={(event) => { const next = event.target.value as ControlMode; setModeValue(next); void runRequest(() => setMode(next)); }}>
                      <option value="fixed">固定导纳</option>
                      <option value="rl">RL 参数调节</option>
                    </select>
                  </div>
                  <div className="home-task-brief">
                    <span className="brief-label">当前任务</span>
                    <strong><TaskGlyph name={task} /> {taskLabels[task]}</strong>
                    <small>{taskMeta[task].category} · {taskMeta[task].level}</small>
                  </div>
                </div>
                <div className="checkin-panel">
                  <div className="checkin-head">
                    <strong>训练前状态</strong>
                    <span>疼痛或疲劳偏高时，Agent 会自动降低建议强度</span>
                  </div>
                  <div className="checkin-grid">
                    <CheckInControl label="疼痛 VAS" value={checkIn.pain_vas} onChange={(pain_vas) => setCheckIn((previous) => ({ ...previous, pain_vas }))} />
                    <CheckInControl label="疲劳程度" value={checkIn.fatigue_0_10} onChange={(fatigue_0_10) => setCheckIn((previous) => ({ ...previous, fatigue_0_10 }))} />
                    <CheckInControl label="当前用力感" value={checkIn.exertion_rpe} onChange={(exertion_rpe) => setCheckIn((previous) => ({ ...previous, exertion_rpe }))} />
                  </div>
                  <input
                    className="checkin-note"
                    value={checkIn.note}
                    onChange={(event) => setCheckIn((previous) => ({ ...previous, note: event.target.value }))}
                    placeholder="可选：记录肩痛、睡眠、情绪或其他当日情况"
                    maxLength={300}
                  />
                </div>
                <button className="primary start-button" disabled={checkIn.pain_vas >= 7} onClick={() => void handleStart("home")}>
                  <PlayGlyph /> {checkIn.pain_vas >= 7 ? "疼痛评分过高，请先复核" : "开始训练"}
                </button>
              </section>

              <section className="dispatch-panel card">
                <div className="patient-panel-head">
                  <h3>任务发布</h3>
                  <small>TASK DISPATCH</small>
                </div>
                <div className="dispatch-body">
                  <div className="dispatch-current">
                    <span>派发内容</span>
                    <strong><TaskGlyph name={task} /> {taskLabels[task]}</strong>
                    <small>沿用当前训练设置的任务与参数</small>
                  </div>
                  <div className="control-field">
                    <label>截止日期</label>
                    <input type="date" value={dueDate} onChange={(event) => setDueDate(event.target.value)} />
                  </div>
                  <button className="primary dispatch-button" disabled={!loadedPatient} onClick={() => void handleDispatch()}>
                    {loadedPatient ? `派发任务给 ${loadedPatient.patient_id}` : "请先加载患者档案"}
                  </button>
                  {dispatched && <div className="dispatch-note">{dispatched}</div>}
                  {assignments.length > 0 && (
                    <div className="dispatch-list">
                      <div className="quicklist-title">派发记录</div>
                      {assignments.slice(0, 6).map((item) => (
                        <div key={item.assignment_id} className={`dispatch-item status-${item.status}`}>
                          <strong>{taskLabels[item.task]}</strong>
                          <span>{item.due_date || "无截止"}</span>
                          <em>{item.status === "completed" ? "已完成" : "未完成"}</em>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </section>
            </div>

            <div className="home-main-col">
              <section className="board-section">
                <div className="section-head">
                  <h3>患者总览</h3>
                  <small>{boardPatients.length} 位在档 · 按最近训练排序</small>
                </div>
                {boardPatients.length > 0 ? (
                  <div className="patient-board">
                    {boardPatients.map((item) => {
                      const avg = averageScore(item.history, 3);
                      const sparkValues = item.history.slice(-8).map((entry) => entry.score);
                      const previousAvg = averageScore(item.history.slice(0, -3), 3);
                      const trendUp = avg !== null && previousAvg !== null && avg > previousAvg + 0.05;
                      const trendDown = avg !== null && previousAvg !== null && avg < previousAvg - 0.05;
                      const active = loadedPatient?.patient_id === item.patient_id;
                      return (
                        <div key={item.patient_id} className={`patient-board-card card ${active ? "active" : ""}`}>
                          <div className="board-card-top">
                            <strong>{item.patient_id}</strong>
                            <span className="board-profile">{patientLabels[item.profile]}</span>
                            {active && <span className="board-active-tag">当前</span>}
                          </div>
                          <div className="board-card-metrics">
                            <div className="board-metric"><span>训练</span><strong>{item.session_count} 次</strong></div>
                            <div className="board-metric"><span>近 3 次均分</span><strong>{avg !== null ? avg.toFixed(2) : "—"}</strong></div>
                            <div className="board-metric"><span>最近训练</span><strong>{formatWhen(item.last_session_at)}</strong></div>
                          </div>
                          <div className={`board-card-bottom ${trendUp ? "trend-up" : trendDown ? "trend-down" : ""}`}>
                            <Sparkline values={sparkValues} />
                            <span className="trend-label">
                              {sparkValues.length < 2 ? "数据不足" : trendUp ? "▲ 好转" : trendDown ? "▼ 下滑" : "— 持平"}
                            </span>
                            <button
                              className="secondary board-open"
                              disabled={patientLoading}
                              onClick={() => { setPatientId(item.patient_id); void loadPatientById(item.patient_id); }}
                            >
                              打开档案
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="board-empty card">暂无建档患者。在左侧输入患者 ID 开始第一位患者的训练，新患者会自动建档。</div>
                )}
              </section>

              <section className="board-section">
                <div className="section-head">
                  <h3>训练任务选择</h3>
                  <small>当前：{taskLabels[task]} · {taskMeta[task].category} · {taskMeta[task].level}</small>
                </div>
                <div className="task-category-tabs" aria-label="按治疗目标筛选任务">
                  {taskCategories.map((category) => (
                    <button
                      key={category}
                      className={taskCategory === category ? "active" : ""}
                      onClick={() => setTaskCategory(category)}
                    >
                      {category}
                    </button>
                  ))}
                </div>
              </section>

              <section className="task-grid">
                {visibleTasks.map((name) => {
                  const meta = taskMeta[name];
                  return (
                    <button
                      key={name}
                      className={`task-card ${task === name ? "selected" : ""}`}
                      onClick={() => handleTaskSelect(name)}
                    >
                      <div className="task-card-top">
                        <span className="task-icon"><TaskGlyph name={name} /></span>
                        <strong>{taskLabels[name]}</strong>
                        <span className="task-check">{task === name ? "● 已选择" : "○ 点击选择"}</span>
                      </div>
                      <div className="task-card-meta">
                        <span className="task-category">{meta.category}</span>
                        <span className={`task-level level-${meta.level}`}>{meta.level}</span>
                      </div>
                      <p>{meta.desc}</p>
                    </button>
                  );
                })}
              </section>

              {taskSpecs.length > 0 && (
                <section className="task-params card">
                  <div className="task-params-head">
                    <h3>任务参数调节</h3>
                    <small>难度 / 速度 / 时长按需调整，仅对本次训练生效</small>
                  </div>
                  <div className="task-params-grid">
                    {taskSpecs.map((spec) => (
                      <TaskParamControl
                        key={spec.name}
                        spec={spec}
                        value={taskParams[spec.name]}
                        onChange={(value) => setTaskParams((previous) => ({ ...previous, [spec.name]: value }))}
                      />
                    ))}
                  </div>
                </section>
              )}
            </div>
          </div>

          <footer className="home-footer">
            <span>PLANAR REHAB CONSOLE · v0.1.0</span>
            <span>安全监督 SafetySupervisor · 参数边界实时裁剪</span>
            <span>SIMULATION ONLY · 遥测 20 Hz</span>
          </footer>
        </div>
      </main>
    );
  }

  if (view === "patient") {
    const tasks = config?.tasks ?? (Object.keys(taskLabels) as TaskName[]);
    const pending = assignments.filter((item) => item.status === "pending");
    const doneCount = assignments.length - pending.length;
    return (
      <main className="app-shell">
        {topbar}
        {error && <div className="error-banner"><span className="alert-icon">!</span><span>{error}</span></div>}

        <div className="home-layout">
          <section className="home-hero">
            <div className="hero-copy">
              <h2>完成今天的训练<em>，离康复更近一步。</em></h2>
              <p>执行医生派发的任务，或选择自主练习；训练数据会自动回传至您的档案，医生将基于数据评估康复效果并优化计划。</p>
            </div>
          </section>

          <section className="patient-panel card">
            <div className="patient-panel-head">
              <h3>患者登录</h3>
              <small>PATIENT LOGIN</small>
            </div>
            <div className="patient-id-row">
              <input
                value={patientId}
                onChange={(event) => setPatientId(event.target.value)}
                onKeyDown={(event) => { if (event.key === "Enter") handleLoadPatient(); }}
                placeholder="请输入您的患者 ID"
                maxLength={32}
              />
              <button className="secondary action-button" onClick={handleLoadPatient} disabled={patientLoading}>
                {patientLoading ? "加载中…" : "加载我的档案"}
              </button>
            </div>
            {loadedPatient && (
              <div className="patient-summary">
                <div className="patient-summary-item"><span>档案</span><strong>{loadedPatient.patient_id} · {patientLabels[loadedPatient.profile]}</strong></div>
                <div className="patient-summary-item"><span>累计训练</span><strong>{loadedPatient.session_count} 次</strong></div>
                <div className="patient-summary-item"><span>上次训练</span><strong>{loadedPatient.last_session_at ? new Date(loadedPatient.last_session_at * 1000).toLocaleString("zh-CN") : "尚无记录"}</strong></div>
              </div>
            )}
          </section>

          <section className="board-section">
            <div className="section-head">
              <h3>任务清单</h3>
              <small>医生派发 · {pending.length} 项未完成 · {doneCount} 项已完成</small>
            </div>
            {pending.length > 0 ? (
              <div className="assignment-board">
                {pending.map((item) => (
                  <div key={item.assignment_id} className="assignment-card card">
                    <div className="board-card-top">
                      <span className="task-icon"><TaskGlyph name={item.task} /></span>
                      <strong>{taskLabels[item.task]}</strong>
                      <span className="assignment-due">截止 {item.due_date || "未设置"}</span>
                    </div>
                    <div className="assignment-meta">
                      <span>{taskMeta[item.task].category} · {taskMeta[item.task].level}</span>
                      <span>{taskMeta[item.task].desc}</span>
                    </div>
                    <button className="primary assignment-enter" onClick={() => void enterAssignment(item)}>
                      <PlayGlyph /> 进入任务
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="board-empty card">暂无派发任务。您可以在下方选择自主练习，或联系医生为您安排训练任务。</div>
            )}
          </section>

          <section className="board-section">
            <div className="section-head">
              <h3>自主练习</h3>
              <small>自由选择任务 · 数据同样回传档案</small>
            </div>
            <div className="task-grid">
              {tasks.map((name) => {
                const meta = taskMeta[name];
                return (
                  <button
                    key={name}
                    className={`task-card ${task === name ? "selected" : ""}`}
                    onClick={() => handleTaskSelect(name)}
                  >
                    <div className="task-card-top">
                      <span className="task-icon"><TaskGlyph name={name} /></span>
                      <strong>{taskLabels[name]}</strong>
                      <span className="task-check">{task === name ? "● 已选择" : "○ 点击选择"}</span>
                    </div>
                    <div className="task-card-meta">
                      <span className="task-category">{meta.category}</span>
                      <span className={`task-level level-${meta.level}`}>{meta.level}</span>
                    </div>
                    <p>{meta.desc}</p>
                  </button>
                );
              })}
            </div>
            <button className="primary start-button self-start" onClick={() => void handleStart("patient")}>
              <PlayGlyph /> 开始自主练习
            </button>
          </section>

          <footer className="home-footer">
            <span>PLANAR REHAB · PATIENT TERMINAL</span>
            <span>执行任务 → 数据回传 → 医生评估 → 计划优化</span>
            <span>SIMULATION ONLY · 遥测 20 Hz</span>
          </footer>
        </div>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      {topbar}
      {error && <div className="error-banner"><span className="alert-icon">!</span><span>{error}</span></div>}

      <div className="training-layout">
        <section className="training-main">
          <div className="training-status-row">
            <button className="ghost-button" onClick={() => finished && setView(originView)} disabled={!finished}>
              ← 返回{finished ? "" : "（训练结束后）"}
            </button>
            <div className="session-brief">
              <strong><TaskGlyph name={snapshot.task} /> {taskLabels[snapshot.task]}</strong>
              <span>{snapshot.patient_id} · {patientLabels[snapshot.patient_profile]} · {modeLabel} · {stateLabels[snapshot.state]}</span>
            </div>
            <div className={`safety-chip ${safetyStatus}`}>
              <span className="status-dot" />{safetyLabel}
            </div>
            <div className="session-buttons">
              {(running || paused) && (
                <button className="danger estop-button" title="立即停止训练并进入安全状态" onClick={() => void runRequest(stopSession)}>
                  ■ 急停
                </button>
              )}
              {running && <button className="action-button pause-button" onClick={() => void runRequest(pauseSession)}>Ⅱ 暂停</button>}
              {paused && <button className="primary action-button" onClick={() => void runRequest(resumeSession)}>▶ 继续</button>}
            </div>
          </div>

          <div className="metric-row">
            <Metric compact glyph="progress" accent="cyan" label="任务进度" value={`${progressPercent}%`} progress={snapshot.task_progress} />
            <Metric compact glyph="score" accent="violet" label="主动参与" value={`${((telemetry?.active_participation_ratio ?? 0) * 100).toFixed(0)}%`} progress={telemetry?.active_participation_ratio ?? 0} />
            <Metric compact glyph="force" accent="orange" label="交互力幅值" value={`${forceMagnitude.toFixed(3)} N`} />
            <Metric compact glyph="fatigue" accent="green" label="训练负荷估计" value={`${((telemetry?.fatigue ?? 0) * 100).toFixed(0)}%`} progress={telemetry?.fatigue ?? 0} />
          </div>

          <div className="chart-grid">
            <Panel className="main-viz-panel" title={taskLabels[snapshot.task]} subtitle="TASK VISUALIZATION" badge={`${history.length} SAMPLES`}>
              {snapshot.task === "maze_navigation" ? <MazeChart points={history} />
                : snapshot.task === "color_memory" ? <ColorMemoryPanel points={history} />
                : <>
                    {snapshot.task === "marker_memory" && telemetry?.task_phase && (
                      <div className="memory-sequence-bar">
                        <span className={`memory-phase phase-${telemetry.task_phase}`}>
                          {telemetry.task_phase === "memorize" ? "记忆标记位置" : "移动到记忆位置"}
                        </span>
                        <span className="memory-hint">回忆阶段标记隐藏，凭记忆到达</span>
                      </div>
                    )}
                    <TrajectoryChart points={history} targets={chartTargets} />
                  </>}
              {agentEvent && (
                <div className={`inline-agent-banner severity-${agentEvent.severity}`}>
                  <span className="agent-tag">AI</span>
                  <span className="inline-agent-message">{agentEvent.message}</span>
                </div>
              )}
            </Panel>
            <Panel title="交互力曲线" subtitle="Fx / Fy / Tz" badge={telemetry ? "LIVE" : "STANDBY"}>
              <LineChart values={forceSeries} height={108} colors={["#4ed7ee", "#ffac63", "#a98aff"]} labels={["Fx", "Fy", "Tz"]} />
            </Panel>
            <Panel title="导纳参数" subtitle="LOW-FREQ UPDATE" badge={modeLabel}>
              <LineChart values={parameterSeries} height={108} colors={["#4ed7ee", "#5ee7a4", "#ffac63", "#a98aff", "#f27caa"]} labels={["Dx", "Dy", "Dθ", "Ka", "λv"]} />
            </Panel>
          </div>

          {snapshot.report && snapshot.agent_summary && (
            <div className="summary-banner card">
              <div className="summary-icon"><StarGlyph /></div>
              <div className="summary-body">
                <strong>{snapshot.agent_summary.message}</strong>
                <span>{snapshot.agent_summary.recommendation}</span>
                <small>{snapshot.agent_summary.highlights.join(" · ")} · 来源：{snapshot.agent_summary.source === "llm" ? "LLM 生成" : "规则模板"}</small>
              </div>
              <button className="primary action-button" onClick={() => { void runRequest(stopSession); setView(originView); }}>完成并返回</button>
            </div>
          )}
        </section>

        <aside className="chat-panel card">
          <div className="chat-header">
            <h2>智能教练</h2>
            <span className="panel-badge">临床辅助 Agent</span>
          </div>
          <div className="chat-feed" ref={chatFeedRef}>
            {chatFeed.length === 0 ? (
              <div className="chat-empty"><div className="empty-icon"><StarGlyph /></div><div><strong>对话尚未开始</strong><span>训练提示与 LLM 反馈会出现在这里，可随时向教练提问。</span></div></div>
            ) : chatFeed.map((item: AgentChatMessage, index: number) => (
              <div className={`chat-message ${item.role} source-${item.source}`} key={`${item.timestamp_s}-${index}`}>
                <span className="chat-avatar">{item.role === "user" ? "你" : "AI"}</span>
                <div className="chat-bubble-wrap">
                  <span className="chat-bubble">{item.message}</span>
                  <small>{item.source === "llm" ? "LLM 生成" : item.source === "rules" ? "规则反馈" : "患者 / 治疗师"}</small>
                </div>
              </div>
            ))}
          </div>
          <div className="chat-input-row">
            <input
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter") void sendChat(); }}
              placeholder="问问智能教练，例如：今天练得怎么样？"
              disabled={chatSending}
            />
            <button className="primary chat-send-button" onClick={() => void sendChat()} disabled={chatSending}>
              ➤ {chatSending ? "思考中" : "发送"}
            </button>
          </div>
        </aside>
      </div>
    </main>
  );
}

function ClinicalProfileCard({
  value,
  editing,
  saving,
  onEdit,
  onCancel,
  onChange,
  onSave
}: {
  value: PatientClinicalProfile;
  editing: boolean;
  saving: boolean;
  onEdit: () => void;
  onCancel: () => void;
  onChange: (value: PatientClinicalProfile) => void;
  onSave: () => void;
}) {
  const affectedLabels = { left: "左侧", right: "右侧", bilateral: "双侧", unknown: "待评估" };
  const stageLabels = { acute: "急性期", subacute: "亚急性期", chronic: "慢性期", unknown: "待评估" };
  const setField = <K extends keyof PatientClinicalProfile>(key: K, next: PatientClinicalProfile[K]) => {
    onChange({ ...value, [key]: next });
  };
  if (!editing) {
    const scoreEntries = Object.entries(value.standardized_scores);
    return (
      <section className="clinical-profile card">
        <div className="patient-panel-head">
          <h3>临床档案</h3>
          <button className="text-button" onClick={onEdit}>编辑</button>
        </div>
        <div className="clinical-summary-grid">
          <div><span>诊断</span><strong>{value.diagnosis || "待补充"}</strong></div>
          <div><span>患侧 / 阶段</span><strong>{affectedLabels[value.affected_side]} · {stageLabels[value.rehab_stage]}</strong></div>
          <div><span>康复目标</span><strong>{value.goals.join("；") || "待补充"}</strong></div>
          <div><span>量表</span><strong>{scoreEntries.length ? scoreEntries.map(([name, score]) => `${name} ${score}`).join(" · ") : "待补充"}</strong></div>
        </div>
        {value.precautions.length > 0 && <div className="clinical-alert">注意：{value.precautions.join("；")}</div>}
      </section>
    );
  }
  return (
    <section className="clinical-profile clinical-edit card">
      <div className="patient-panel-head"><h3>编辑临床档案</h3><small>由医生维护</small></div>
      <div className="clinical-form-grid">
        <label><span>诊断</span><input value={value.diagnosis} onChange={(event) => setField("diagnosis", event.target.value)} placeholder="如：脑卒中后上肢功能障碍" /></label>
        <label><span>发病日期</span><input type="date" value={value.onset_date} onChange={(event) => setField("onset_date", event.target.value)} /></label>
        <label><span>患侧</span><select value={value.affected_side} onChange={(event) => setField("affected_side", event.target.value as PatientClinicalProfile["affected_side"])}><option value="unknown">待评估</option><option value="left">左侧</option><option value="right">右侧</option><option value="bilateral">双侧</option></select></label>
        <label><span>康复阶段</span><select value={value.rehab_stage} onChange={(event) => setField("rehab_stage", event.target.value as PatientClinicalProfile["rehab_stage"])}><option value="unknown">待评估</option><option value="acute">急性期</option><option value="subacute">亚急性期</option><option value="chronic">慢性期</option></select></label>
        <label><span>FMA-UE</span><input type="number" min="0" max="66" value={value.standardized_scores["FMA-UE"] ?? ""} onChange={(event) => setField("standardized_scores", { ...value.standardized_scores, "FMA-UE": Number(event.target.value) })} /></label>
        <label><span>ARAT</span><input type="number" min="0" max="57" value={value.standardized_scores.ARAT ?? ""} onChange={(event) => setField("standardized_scores", { ...value.standardized_scores, ARAT: Number(event.target.value) })} /></label>
      </div>
      <label className="clinical-wide"><span>康复目标（每行一项）</span><textarea value={value.goals.join("\n")} onChange={(event) => setField("goals", splitLines(event.target.value))} placeholder="例如：提高患侧手臂主动到达能力" /></label>
      <label className="clinical-wide"><span>注意事项 / 禁忌（每行一项）</span><textarea value={value.precautions.join("\n")} onChange={(event) => setField("precautions", splitLines(event.target.value))} placeholder="例如：肩痛，避免超过舒适活动范围" /></label>
      <label className="clinical-wide"><span>补充记录</span><textarea value={value.notes} onChange={(event) => setField("notes", event.target.value)} /></label>
      <div className="clinical-actions"><button className="secondary" onClick={onCancel}>取消</button><button className="primary" disabled={saving} onClick={onSave}>{saving ? "保存中…" : "保存临床档案"}</button></div>
    </section>
  );
}

function CheckInControl({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="checkin-control">
      <span>{label}</span>
      <div><input type="range" min="0" max="10" step="1" value={value} onChange={(event) => onChange(Number(event.target.value))} /><strong>{value}/10</strong></div>
    </label>
  );
}

function splitLines(value: string): string[] {
  return value.split(/\n|；/).map((item) => item.trim()).filter(Boolean);
}

function TaskParamControl({ spec, value, onChange }: { spec: TaskParamSpec; value: number | string | undefined; onChange: (value: number | string) => void }) {
  const current = value ?? spec.default ?? (spec.type === "select" ? (spec.options?.[0] ?? "") : spec.min ?? 0);
  if (spec.type === "select") {
    return (
      <div className="control-field task-param-field">
        <label>{spec.label}</label>
        <select value={String(current)} onChange={(event) => onChange(event.target.value)}>
          {spec.options?.map((option) => (
            <option key={String(option)} value={String(option)}>{spec.labels?.[String(option)] ?? option}{spec.unit ? ` ${spec.unit}` : ""}</option>
          ))}
        </select>
      </div>
    );
  }
  const numeric = Number(current);
  return (
    <div className="control-field task-param-field">
      <label>{spec.label}{spec.unit ? `（${spec.unit}）` : ""}</label>
      <div className="slider-row">
        <input
          type="range"
          min={spec.min ?? 0}
          max={spec.max ?? 1}
          step={spec.step ?? 0.01}
          value={numeric}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <span className="slider-value">{numeric.toFixed((spec.step ?? 0.01) < 0.01 ? 2 : 2)}</span>
      </div>
    </div>
  );
}

const glyphProps = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const
};

function TaskGlyph({ name }: { name: TaskName }) {
  switch (name) {
    case "point_to_point":
      return (
        <svg viewBox="0 0 24 24" width="18" height="18" {...glyphProps} aria-hidden>
          <circle cx="5.2" cy="12" r="2.8" />
          <line x1="9.4" y1="12" x2="16.8" y2="12" />
          <circle cx="19" cy="12" r="2.8" fill="currentColor" stroke="none" />
        </svg>
      );
    case "circle_tracking":
      return (
        <svg viewBox="0 0 24 24" width="18" height="18" {...glyphProps} aria-hidden>
          <circle cx="12" cy="12" r="7.6" />
          <path d="M15.2 5.6 A 7.6 7.6 0 0 1 19.6 12" />
          <polyline points="16.6,8.2 19.6,12 15.6,12.2" />
        </svg>
      );
    case "figure8_tracking":
      return (
        <svg viewBox="0 0 24 24" width="18" height="18" {...glyphProps} aria-hidden>
          <circle cx="9.6" cy="12" r="4.9" />
          <circle cx="14.4" cy="12" r="4.9" />
        </svg>
      );
    case "maze_navigation":
      return (
        <svg viewBox="0 0 24 24" width="18" height="18" {...glyphProps} aria-hidden>
          <rect x="3.6" y="3.6" width="16.8" height="16.8" rx="2.6" />
          <path d="M8.4 3.6 V 10 M12.6 14 V 20.4 M3.6 13.8 H 9.8 M15.2 10.2 H 20.4" />
        </svg>
      );
    case "color_memory":
      return (
        <svg viewBox="0 0 24 24" width="18" height="18" {...glyphProps} aria-hidden>
          <rect x="4.6" y="4.6" width="6.2" height="6.2" rx="1.6" />
          <rect x="13.2" y="4.6" width="6.2" height="6.2" rx="1.6" />
          <rect x="4.6" y="13.2" width="6.2" height="6.2" rx="1.6" />
          <rect x="13.2" y="13.2" width="6.2" height="6.2" rx="1.6" />
        </svg>
      );
    case "follow_to_reach":
      return (
        <svg viewBox="0 0 24 24" width="18" height="18" {...glyphProps} aria-hidden>
          <circle cx="12" cy="12" r="8.2" />
          <circle cx="12" cy="12" r="4.4" strokeDasharray="3 2.4" />
          <circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none" />
        </svg>
      );
    case "visual_guided_reach":
      return (
        <svg viewBox="0 0 24 24" width="18" height="18" {...glyphProps} aria-hidden>
          <path d="M2.8 12 C 6 6.8 18 6.8 21.2 12 C 18 17.2 6 17.2 2.8 12 Z" />
          <circle cx="12" cy="12" r="2.6" />
        </svg>
      );
    case "motion_intercept":
      return (
        <svg viewBox="0 0 24 24" width="18" height="18" {...glyphProps} aria-hidden>
          <circle cx="4.8" cy="12" r="1.9" fill="currentColor" stroke="none" />
          <circle cx="19.2" cy="12" r="1.9" />
          <path d="M8 9.4 H 15.2" />
          <polyline points="13,7.2 15.2,9.4 13,11.6" />
          <path d="M16 14.6 H 8.8" />
          <polyline points="11,12.4 8.8,14.6 11,16.8" />
        </svg>
      );
    case "marker_memory":
      return (
        <svg viewBox="0 0 24 24" width="18" height="18" {...glyphProps} aria-hidden>
          <rect x="4.2" y="4.2" width="15.6" height="15.6" rx="2.4" strokeDasharray="3.2 2.6" />
          <circle cx="14.6" cy="9.6" r="2.2" fill="currentColor" stroke="none" />
        </svg>
      );
  }
}

function MetricGlyph({ name }: { name: string }) {
  const props = { ...glyphProps, width: 13, height: 13 };
  switch (name) {
    case "progress":
      return (
        <svg viewBox="0 0 24 24" {...props} aria-hidden>
          <circle cx="12" cy="12" r="8.6" />
          <path d="M12 3.4 A 8.6 8.6 0 0 1 20.6 12" />
        </svg>
      );
    case "score":
      return (
        <svg viewBox="0 0 24 24" {...props} aria-hidden>
          <path d="M12 3.4 L14.2 9.2 L20.4 9.9 L15.6 14.1 L17.1 20.2 L12 16.9 L6.9 20.2 L8.4 14.1 L3.6 9.9 L9.8 9.2 Z" />
        </svg>
      );
    case "force":
      return (
        <svg viewBox="0 0 24 24" {...props} aria-hidden>
          <path d="M3 12 C5.2 7.4 7.4 7.4 9.6 12 C11.8 16.6 14 16.6 16.2 12 C18.4 7.4 20.6 7.4 21.6 12" />
        </svg>
      );
    case "fatigue":
      return (
        <svg viewBox="0 0 24 24" {...props} aria-hidden>
          <path d="M3 12.4 H8 L10.2 7.6 L13.4 15.6 L15.6 10.8 H21" />
        </svg>
      );
  }
}

function PlayGlyph() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden>
      <path d="M7.5 4.8 L19 12 L7.5 19.2 Z" />
    </svg>
  );
}

function StarGlyph() {
  return (
    <svg viewBox="0 0 24 24" width="17" height="17" {...glyphProps} aria-hidden>
      <path d="M12 3.4 L14.2 9.2 L20.4 9.9 L15.6 14.1 L17.1 20.2 L12 16.9 L6.9 20.2 L8.4 14.1 L3.6 9.9 L9.8 9.2 Z" />
    </svg>
  );
}

function Metric({ glyph, accent, label, value, progress, compact = false }: { glyph: string; accent: string; label: string; value: string; progress?: number; compact?: boolean }) {
  return (
    <div className={`metric-card card accent-${accent} ${compact ? "metric-compact" : ""}`}>
      <div className="metric-top">
        <span className="metric-icon"><MetricGlyph name={glyph} /></span>
        <span className="metric-label">{label}</span>
        <span className="metric-live">LIVE</span>
      </div>
      <strong>{value}</strong>
      {progress !== undefined && <div className="progress-track"><div style={{ transform: `scaleX(${progress})` }} /></div>}
    </div>
  );
}

function Panel({ title, subtitle, badge, className = "", children }: { title: string; subtitle: string; badge?: string; className?: string; children: ReactNode }) {
  return (
    <div className={`card panel ${className}`}>
      <div className="panel-heading">
        <h2>{title}</h2>
        {badge && <span className="panel-badge">{badge}</span>}
      </div>
      {children}
    </div>
  );
}

function formatWhen(ts: number | null): string {
  if (!ts) return "暂无记录";
  const date = new Date(ts * 1000);
  const now = new Date();
  const time = date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  if (date.toDateString() === now.toDateString()) return `今天 ${time}`;
  return `${date.getMonth() + 1}/${date.getDate()} ${time}`;
}

function averageScore(history: { score: number }[], count: number): number | null {
  const recent = history.slice(-count);
  if (recent.length === 0) return null;
  return recent.reduce((sum, entry) => sum + entry.score, 0) / recent.length;
}

function Sparkline({ values }: { values: number[] }) {
  const width = 92;
  const height = 30;
  if (values.length < 2) {
    return (
      <svg className="board-spark" width={width} height={height} aria-hidden>
        <line x1="2" y1={height / 2} x2={width - 2} y2={height / 2} className="spark-flat" />
      </svg>
    );
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pad = 4;
  const points = values.map((value, index) => {
    const x = pad + (index / (values.length - 1)) * (width - pad * 2);
    const y = height - pad - ((value - min) / range) * (height - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const [lastX, lastY] = points[points.length - 1].split(",").map(Number);
  return (
    <svg className="board-spark" width={width} height={height} aria-hidden>
      <polyline points={points.join(" ")} className="spark-line" />
      <circle cx={lastX} cy={lastY} r="2.4" className="spark-dot" />
    </svg>
  );
}

export default App;
