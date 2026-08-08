import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { LineChart } from "./components/LineChart";
import { TrajectoryChart } from "./components/TrajectoryChart";
import { MazeChart } from "./components/MazeChart";
import { ColorMemoryPanel } from "./components/ColorMemoryPanel";
import {
  chatWithAgent,
  getConfig,
  getSession,
  pauseSession,
  resumeSession,
  setMode,
  startSession,
  stopSession
} from "./services/api";
import type {
  AgentChatMessage,
  ConfigSummary,
  ControlMode,
  PatientProfile,
  SessionSnapshot,
  TaskName,
  Telemetry
} from "./types";
import "./styles.css";

const taskLabels: Record<TaskName, string> = {
  point_to_point: "点到点训练",
  circle_tracking: "圆轨迹训练",
  figure8_tracking: "八字轨迹训练",
  maze_navigation: "迷宫导航",
  color_memory: "色块记忆"
};

const taskMeta: Record<TaskName, { icon: string; category: string; desc: string; level: string }> = {
  point_to_point: { icon: "⌁", category: "轨迹跟踪", desc: "从起点平稳移动至目标点，训练运动启动与定位控制能力。", level: "基础" },
  circle_tracking: { icon: "◍", category: "轨迹跟踪", desc: "沿圆形参考轨迹连续运动，训练节律性与速度适应能力。", level: "进阶" },
  figure8_tracking: { icon: "∞", category: "轨迹跟踪", desc: "沿八字轨迹换向跟踪，训练方向切换与协调控制。", level: "进阶" },
  maze_navigation: { icon: "▦", category: "空间导航", desc: "引导末端穿越 S 形迷宫走廊，训练空间规划与运动协调能力。", level: "挑战" },
  color_memory: { icon: "▩", category: "认知训练", desc: "记忆颜色序列并按顺序复述，记忆与运动的双任务训练。", level: "挑战" }
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

const initialSnapshot: SessionSnapshot = {
  session_id: "none",
  state: "idle",
  task: "point_to_point",
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
  const [view, setView] = useState<"home" | "training">("home");
  const [task, setTask] = useState<TaskName>("point_to_point");
  const [patient, setPatient] = useState<PatientProfile>("moderate");
  const [mode, setModeValue] = useState<ControlMode>("fixed");
  const [error, setError] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [chatSending, setChatSending] = useState(false);
  const chatFeedRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void Promise.all([getConfig(), getSession()])
      .then(([loadedConfig, loadedSession]) => {
        setConfig(loadedConfig);
        setSnapshot(loadedSession);
        setModeValue(loadedSession.mode);
        if (loadedSession.state === "running" || loadedSession.state === "paused") {
          setView("training");
        }
      })
      .catch((reason: unknown) => setError(String(reason)));

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws/telemetry`);
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data) as { type: string; data: SessionSnapshot };
      if (message.type !== "telemetry") return;
      setError((previous) => (previous && previous.includes("WebSocket") ? null : previous));
      setSnapshot(message.data);
      if (message.data.telemetry) {
        setHistory((previous) => [...previous, message.data.telemetry as Telemetry].slice(-240));
      }
    };
    socket.onerror = () => setError("WebSocket 连接失败，请确认 FastAPI 后端已启动");
    return () => socket.close();
  }, []);

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

  const handleStart = async () => {
    const next = await runRequest(() => startSession(task, patient, mode));
    if (next) setView("training");
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

  const chatFeed = snapshot.agent_chat ?? [];

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

  const topbar = (
    <header className="topbar">
      <div className="brand-lockup">
        <div className="brand-emblem"><span>R</span><i /></div>
        <div>
          <p className="eyebrow"><span className="eyebrow-line" /> REHAB / INTELLIGENCE SYSTEM</p>
          <h1>上肢康复训练<span>控制台</span></h1>
        </div>
      </div>
      <div className="topbar-actions">
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
    return (
      <main className="app-shell">
        <div className="ambient ambient-one" />
        <div className="ambient ambient-two" />
        {topbar}
        {error && <div className="error-banner"><span className="alert-icon">!</span><span>{error}</span></div>}

        <div className="home-layout">
          <section className="home-hero">
            <div className="live-label"><span className="live-pulse" /> TRAINING TASK LIBRARY</div>
            <h2>选择一项训练任务<em>，开始今天的康复。</em></h2>
            <p>覆盖轨迹跟踪、空间导航与认知双任务训练；安全强化学习实时调节导纳参数，智能教练全程陪伴。</p>
          </section>

          <section className="task-grid">
            {tasks.map((name) => {
              const meta = taskMeta[name];
              return (
                <button
                  key={name}
                  className={`task-card ${task === name ? "selected" : ""}`}
                  onClick={() => setTask(name)}
                >
                  <div className="task-card-top">
                    <span className="task-icon">{meta.icon}</span>
                    <span className="task-category">{meta.category}</span>
                    <span className={`task-level level-${meta.level}`}>{meta.level}</span>
                  </div>
                  <strong>{taskLabels[name]}</strong>
                  <p>{meta.desc}</p>
                  <span className="task-check">{task === name ? "● 已选择" : "○ 点击选择"}</span>
                </button>
              );
            })}
          </section>

          <section className="home-controls card">
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
                <strong>{taskMeta[task].icon} {taskLabels[task]}</strong>
                <small>{taskMeta[task].category} · {taskMeta[task].level}</small>
              </div>
            </div>
            <button className="primary start-button" onClick={() => void handleStart()}>
              <span>▶</span> 开始训练
            </button>
          </section>
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
            <button className="ghost-button" onClick={() => finished && setView("home")} disabled={!finished}>
              ← 返回{finished ? "" : "（训练结束后）"}
            </button>
            <div className="session-brief">
              <strong>{taskMeta[snapshot.task].icon} {taskLabels[snapshot.task]}</strong>
              <span>{patientLabels[snapshot.patient_profile]} · {modeLabel} · {stateLabels[snapshot.state]}</span>
            </div>
            <div className={`safety-chip ${safetyStatus}`}>
              <span className="status-dot" />{safetyLabel}
            </div>
            <div className="session-buttons">
              {running && <button className="action-button pause-button" onClick={() => void runRequest(pauseSession)}>Ⅱ 暂停</button>}
              {paused && <button className="primary action-button" onClick={() => void runRequest(resumeSession)}>▶ 继续</button>}
              {(running || paused) && <button className="danger action-button" onClick={() => void runRequest(stopSession)}>■ 停止</button>}
            </div>
          </div>

          <div className="metric-row">
            <Metric compact icon="◒" accent="cyan" label="任务进度" value={`${progressPercent}%`} progress={snapshot.task_progress} />
            <Metric compact icon="✦" accent="violet" label="当前得分" value={snapshot.score.toFixed(2)} />
            <Metric compact icon="≋" accent="orange" label="交互力幅值" value={`${forceMagnitude.toFixed(3)} N`} />
            <Metric compact icon="♧" accent="green" label="疲劳估计" value={`${((telemetry?.fatigue ?? 0) * 100).toFixed(0)}%`} progress={telemetry ? 1 - telemetry.fatigue : 1} />
          </div>

          <div className="chart-grid">
            <Panel className="main-viz-panel" title={taskLabels[snapshot.task]} subtitle="TASK VISUALIZATION" badge={`${history.length} SAMPLES`}>
              {snapshot.task === "maze_navigation" ? <MazeChart points={history} />
                : snapshot.task === "color_memory" ? <ColorMemoryPanel points={history} />
                : <TrajectoryChart points={history} />}
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
              <div className="summary-icon">✦</div>
              <div className="summary-body">
                <strong>{snapshot.agent_summary.message}</strong>
                <span>{snapshot.agent_summary.recommendation}</span>
                <small>{snapshot.agent_summary.highlights.join(" · ")} · 来源：{snapshot.agent_summary.source === "llm" ? "LLM 生成" : "规则模板"}</small>
              </div>
              <button className="primary action-button" onClick={() => { void runRequest(stopSession); setView("home"); }}>完成并返回</button>
            </div>
          )}
        </section>

        <aside className="chat-panel card">
          <div className="chat-header">
            <div>
              <div className="section-kicker">LLM INTERACTION</div>
              <h2>智能教练</h2>
            </div>
            <span className="panel-badge">DeepSeek</span>
          </div>
          <div className="chat-feed" ref={chatFeedRef}>
            {chatFeed.length === 0 ? (
              <div className="chat-empty"><div className="empty-icon">✦</div><div><strong>对话尚未开始</strong><span>训练提示与 LLM 反馈会出现在这里，可随时向教练提问。</span></div></div>
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

function Metric({ icon, accent, label, value, progress, compact = false }: { icon: string; accent: string; label: string; value: string; progress?: number; compact?: boolean }) {
  return (
    <div className={`metric-card card accent-${accent} ${compact ? "metric-compact" : ""}`}>
      <div className="metric-top">
        <span className="metric-icon">{icon}</span>
        <span className="metric-label">{label}</span>
        <span className="metric-live">LIVE</span>
      </div>
      <strong>{value}</strong>
      {progress !== undefined && <div className="progress-track"><div style={{ width: `${progress * 100}%` }} /></div>}
    </div>
  );
}

function Panel({ title, subtitle, badge, className = "", children }: { title: string; subtitle: string; badge?: string; className?: string; children: ReactNode }) {
  return (
    <div className={`card panel ${className}`}>
      <div className="panel-heading">
        <div><div className="section-kicker">{subtitle}</div><h2>{title}</h2></div>
        {badge && <span className="panel-badge">{badge}</span>}
      </div>
      {children}
    </div>
  );
}

export default App;
