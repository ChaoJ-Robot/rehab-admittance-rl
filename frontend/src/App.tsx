import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { LineChart } from "./components/LineChart";
import { TrajectoryChart } from "./components/TrajectoryChart";
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
  figure8_tracking: "八字轨迹训练"
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
      })
      .catch((reason: unknown) => setError(String(reason)));

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws/telemetry`);
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data) as { type: string; data: SessionSnapshot };
      if (message.type !== "telemetry") return;
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
    } catch (reason) {
      setError(String(reason));
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

  const chatFeed = snapshot.agent_chat ?? [];

  useEffect(() => {
    const feed = chatFeedRef.current;
    if (feed) feed.scrollTop = feed.scrollHeight;
  }, [chatFeed.length]);

  const telemetry = snapshot.telemetry;
  const agentEvent = snapshot.agent_event ?? telemetry?.agent_event ?? null;
  const forceSeries = useMemo(
    () => [0, 1, 2].map((axis) => history.map((point) => point.interaction_force[axis])),
    [history]
  );
  const parameterSeries = useMemo(
    () => [0, 1, 2, 3, 4].map((axis) => history.map((point) => point.admittance_parameters[axis])),
    [history]
  );
  const progressPercent = Math.round(snapshot.task_progress * 100);
  const running = snapshot.state === "running";
  const paused = snapshot.state === "paused";
  const safetyStatus = telemetry?.safety_status ?? "idle";
  const safetyLabel = safetyStatus === "fallback" ? "保护回退" : safetyStatus === "safe" ? "系统安全" : "等待遥测";
  const connectionLabel = error ? "连接异常" : telemetry ? "实时连接" : "等待连接";
  const forceMagnitude = Math.hypot(telemetry?.interaction_force[0] ?? 0, telemetry?.interaction_force[1] ?? 0);
  const modeLabel = snapshot.mode === "rl" ? "RL 参数调节" : "固定导纳";

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

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
          <div className="avatar-mark">3D</div>
        </div>
      </header>

      {error && <div className="error-banner"><span className="alert-icon">!</span><span>{error}</span></div>}

      <section className="hero card">
        <div className="hero-copy">
          <div className="live-label"><span className="live-pulse" /> LIVE SESSION MONITOR</div>
          <h2>让每一次训练，都变得<br /><em>更安全、更主动。</em></h2>
          <p>基于安全强化学习与交互 Agent 的平面三自由度上肢康复训练系统，实时感知轨迹、交互力与患者主动参与度。</p>
          <div className="hero-pills">
            <span><b>01</b> 任务空间控制</span>
            <span><b>02</b> 独立安全监督</span>
            <span><b>03</b> 患者状态反馈</span>
          </div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="radar-ring ring-large" />
          <div className="radar-ring ring-small" />
          <div className="radar-sweep" />
          <div className="robot-orb"><span>3</span><small>DOF</small></div>
          <div className="orbit-tag orbit-top">X / Y <b>▰</b></div>
          <div className="orbit-tag orbit-right">SAFE <b>✓</b></div>
          <div className="orbit-tag orbit-bottom">RL <b>↗</b></div>
          <div className="orbit-node node-one" /><div className="orbit-node node-two" /><div className="orbit-node node-three" />
        </div>
      </section>

      <section className="control-card card">
        <div className="section-identity">
          <span className="section-number">01</span>
          <div><strong>配置训练会话</strong><small>SESSION CONFIGURATION</small></div>
        </div>
        <div className="control-fields">
          <ControlField label="训练任务" icon="⌁">
            <select id="task" value={task} onChange={(event) => setTask(event.target.value as TaskName)} disabled={running || paused}>
              {(config?.tasks ?? ["point_to_point"]).map((item) => <option key={item} value={item}>{taskLabels[item]}</option>)}
            </select>
          </ControlField>
          <ControlField label="患者配置" icon="◉">
            <select id="patient" value={patient} onChange={(event) => setPatient(event.target.value as PatientProfile)} disabled={running || paused}>
              {(config?.patient_profiles ?? ["moderate"]).map((item) => <option key={item} value={item}>{patientLabels[item]}</option>)}
            </select>
          </ControlField>
          <ControlField label="控制模式" icon="ϟ">
            <select id="mode" value={mode} onChange={(event) => { const next = event.target.value as ControlMode; setModeValue(next); void runRequest(() => setMode(next)); }}>
              <option value="fixed">固定导纳</option>
              <option value="rl">RL 参数调节</option>
            </select>
          </ControlField>
        </div>
        <div className="button-row">
          {!running && !paused && <button className="primary action-button" onClick={() => void runRequest(() => startSession(task, patient, mode))}><span>▶</span> 开始训练</button>}
          {running && <button className="action-button pause-button" onClick={() => void runRequest(pauseSession)}><span>Ⅱ</span> 暂停</button>}
          {paused && <button className="primary action-button" onClick={() => void runRequest(resumeSession)}><span>▶</span> 继续训练</button>}
          {(running || paused) && <button className="danger action-button" onClick={() => void runRequest(stopSession)}><span>■</span> 停止</button>}
        </div>
      </section>

      <section className="metric-grid">
        <Metric icon="◒" accent="cyan" label="任务进度" value={`${progressPercent}%`} detail={`${snapshot.elapsed_s.toFixed(1)} / ${snapshot.duration_s.toFixed(1)} s`} progress={snapshot.task_progress} />
        <Metric icon="✦" accent="violet" label="当前得分" value={snapshot.score.toFixed(2)} detail={taskLabels[snapshot.task]} />
        <Metric icon="≋" accent="orange" label="交互力幅值" value={`${forceMagnitude.toFixed(3)} N`} detail="Fx / Fy / Tz" />
        <Metric icon="♧" accent="green" label="患者主动功率" value={`${(telemetry?.human_power_w ?? 0).toFixed(3)} W`} detail={`疲劳 ${(telemetry?.fatigue ?? 0).toFixed(0)}%`} progress={telemetry?.fatigue ? 1 - telemetry.fatigue : 1} />
      </section>

      <section className={`agent-banner card severity-${agentEvent?.severity ?? "info"}`}>
        <div className="agent-mark"><span>AI</span><i /></div>
        <div className="agent-message"><p className="agent-label"><span>INTERACTION AGENT</span> / {agentEvent?.event ?? "SYSTEM STANDBY"}</p><strong>{agentEvent?.message ?? "开始训练后，这里会显示基于运动状态的训练提示。"}</strong></div>
        <div className="agent-side"><span className="agent-side-dot" />只读反馈模式</div>
      </section>
      
      <section className="card agent-chat-card">
        <div className="panel-heading">
          <div><div className="section-kicker">LLM INTERACTION</div><h2>智能教练</h2><p>根据患者表现生成个性化反馈与总结，可随时提问</p></div>
          <span className="panel-badge">DeepSeek</span>
        </div>
        <div className="chat-feed" ref={chatFeedRef}>
          {chatFeed.length === 0 ? (
            <div className="chat-empty"><div className="empty-icon">✦</div><div><strong>对话尚未开始</strong><span>开始训练后，规则事件与 LLM 反馈会出现在这里；训练结束后可向智能教练提问。</span></div></div>
          ) : chatFeed.map((item: AgentChatMessage, index: number) => (
            <div className={`chat-message ${item.role} source-${item.source}`} key={`${item.timestamp_s}-${index}`}>
              <span className="chat-avatar">{item.role === "user" ? "你" : "AI"}</span>
              <div className="chat-bubble-wrap"><span className="chat-bubble">{item.message}</span><small>{item.source === "llm" ? "LLM 生成" : item.source === "rules" ? "规则反馈" : "患者 / 治疗师"}</small></div>
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
          <button className="primary action-button" onClick={() => void sendChat()} disabled={chatSending}><span>➤</span> {chatSending ? "思考中" : "发送"}</button>
        </div>
      </section>
      
      <section className="dashboard-grid">
        <Panel className="trajectory-panel" title="实时轨迹" subtitle="TASK SPACE / [ X, Y ]" badge={`${history.length} SAMPLES`}><TrajectoryChart points={history} /></Panel>
        <Panel title="交互力曲线" subtitle="FORCE TELEMETRY / [ Fx, Fy, Tz ]" badge={telemetry ? "LIVE" : "STANDBY"}><LineChart values={forceSeries} colors={["#4ed7ee", "#ffac63", "#a98aff"]} labels={["Fx", "Fy", "Tz"]} /></Panel>
        <Panel title="导纳参数" subtitle="ADMITTANCE / LOW-FREQUENCY UPDATE" badge={modeLabel}><LineChart values={parameterSeries} colors={["#4ed7ee", "#5ee7a4", "#ffac63", "#a98aff", "#f27caa"]} labels={["Dx", "Dy", "Dθ", "Ka", "λv"]} /></Panel>
        <Panel className="status-panel" title="当前状态" subtitle="SYSTEM TELEMETRY / 20 HZ" badge={safetyLabel}>
          <div className="status-hero">
            <div className={`safety-orb ${safetyStatus}`}><span>{safetyStatus === "fallback" ? "!" : "✓"}</span></div>
            <div><small>SAFETY STATUS</small><strong>{safetyLabel}</strong><em>{snapshot.state === "running" ? "系统正在持续监测" : "等待下一次训练会话"}</em></div>
          </div>
          <div className="state-list">
            <StateRow label="会话状态" value={stateLabels[snapshot.state]} tone={running ? "active" : "normal"} />
            <StateRow label="控制模式" value={modeLabel} />
            <StateRow label="当前位置" value={telemetry ? `[${telemetry.actual_pose.map((value) => value.toFixed(3)).join(", ")}]` : "等待数据"} />
            <StateRow label="当前动作" value={telemetry ? `[${telemetry.rl_action.map((value) => value.toFixed(2)).join(", ")}]` : "[0, 0, 0, 0]"} />
          </div>
        </Panel>
      </section>

      <section className="card report-card">
        <div className="panel-heading"><div><div className="section-kicker">SESSION REPORT</div><h2>训练摘要</h2><p>会话结束后生成的可追溯指标</p></div><span className="session-id">ID / {snapshot.session_id}</span></div>
        {snapshot.report ? <>
          {snapshot.agent_summary && <div className="agent-summary"><div className="summary-icon">✦</div><div><strong>{snapshot.agent_summary.message}</strong><span>{snapshot.agent_summary.recommendation}</span><small>{snapshot.agent_summary.highlights.join(" · ")}</small></div></div>}
          <div className="report-grid">
            <ReportItem label="完成率" value={`${(snapshot.report.completion_rate * 100).toFixed(0)}%`} />
            <ReportItem label="平均轨迹误差" value={snapshot.report.average_tracking_error.toFixed(4)} />
            <ReportItem label="峰值交互力" value={`${snapshot.report.peak_interaction_force.toFixed(4)} N`} />
            <ReportItem label="运动平滑度" value={snapshot.report.motion_smoothness.toFixed(4)} />
            <ReportItem label="患者主动做功" value={`${snapshot.report.patient_active_work.toFixed(4)} J`} />
            <ReportItem label="机器人辅助做功" value={`${snapshot.report.robot_assistance_work.toFixed(4)} J`} />
          </div>
        </> : <div className="empty-report"><div className="empty-icon">⌁</div><div><strong>暂无训练摘要</strong><span>完成一次训练后，这里会显示训练质量和安全指标。</span></div></div>}
      </section>

      <footer><span className="footer-brand"><b>R</b> REHAB INTELLIGENCE</span><span>Phase 7 simulation interface</span><span>RL does not publish motor or joint commands</span><span className="footer-validation">● {config?.hardware_validation_required ? "HARDWARE VALIDATION REQUIRED" : "SIMULATION"}</span></footer>
    </main>
  );
}

function ControlField({ label, icon, children }: { label: string; icon: string; children: ReactNode }) {
  return <div className="control-field"><label><span className="field-icon">{icon}</span>{label}</label>{children}</div>;
}

function Metric({ icon, accent, label, value, detail, progress }: { icon: string; accent: string; label: string; value: string; detail: string; progress?: number }) {
  return <div className={`metric-card card accent-${accent}`}><div className="metric-top"><span className="metric-icon">{icon}</span><span className="metric-label">{label}</span><span className="metric-live">LIVE</span></div><strong>{value}</strong><small>{detail}</small>{progress !== undefined && <div className="progress-track"><div style={{ width: `${progress * 100}%` }} /></div>}</div>;
}

function Panel({ title, subtitle, badge, className = "", children }: { title: string; subtitle: string; badge?: string; className?: string; children: ReactNode }) {
  return <div className={`card panel ${className}`}><div className="panel-heading"><div><div className="section-kicker">{subtitle}</div><h2>{title}</h2></div>{badge && <span className="panel-badge">{badge}</span>}</div>{children}</div>;
}

function StateRow({ label, value, tone = "normal" }: { label: string; value: string; tone?: "normal" | "active" | "warning" }) {
  return <div className="state-row"><span>{label}</span><strong className={`tone-${tone}`}>{tone === "active" && <i />} {value}</strong></div>;
}

function ReportItem({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

export default App;
