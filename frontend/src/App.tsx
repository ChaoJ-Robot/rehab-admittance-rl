import { useEffect, useMemo, useState } from "react";
import { LineChart } from "./components/LineChart";
import { TrajectoryChart } from "./components/TrajectoryChart";
import {
  getConfig,
  getSession,
  pauseSession,
  resumeSession,
  setMode,
  startSession,
  stopSession
} from "./services/api";
import type {
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
  report: null
};

function App() {
  const [config, setConfig] = useState<ConfigSummary | null>(null);
  const [snapshot, setSnapshot] = useState<SessionSnapshot>(initialSnapshot);
  const [history, setHistory] = useState<Telemetry[]>([]);
  const [task, setTask] = useState<TaskName>("point_to_point");
  const [patient, setPatient] = useState<PatientProfile>("moderate");
  const [mode, setModeValue] = useState<ControlMode>("fixed");
  const [error, setError] = useState<string | null>(null);

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

  const telemetry = snapshot.telemetry;
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

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">PLANAR 3-DOF REHABILITATION</p>
          <h1>上肢康复训练控制台</h1>
        </div>
        <div className="status-group">
          <span className={`status-dot ${telemetry?.safety_status ?? "idle"}`} />
          <span>{stateLabels[snapshot.state]}</span>
          <span className="simulation-tag">SIMULATION ONLY</span>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="control-card card">
        <div className="control-field">
          <label htmlFor="task">训练任务</label>
          <select id="task" value={task} onChange={(event) => setTask(event.target.value as TaskName)} disabled={running || paused}>
            {(config?.tasks ?? ["point_to_point"]).map((item) => <option key={item} value={item}>{taskLabels[item]}</option>)}
          </select>
        </div>
        <div className="control-field">
          <label htmlFor="patient">患者配置</label>
          <select id="patient" value={patient} onChange={(event) => setPatient(event.target.value as PatientProfile)} disabled={running || paused}>
            {(config?.patient_profiles ?? ["moderate"]).map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </div>
        <div className="control-field">
          <label htmlFor="mode">控制模式</label>
          <select id="mode" value={mode} onChange={(event) => { const next = event.target.value as ControlMode; setModeValue(next); void runRequest(() => setMode(next)); }}>
            <option value="fixed">固定导纳</option>
            <option value="rl">RL 参数调节</option>
          </select>
        </div>
        <div className="button-row">
          {!running && !paused && <button className="primary" onClick={() => void runRequest(() => startSession(task, patient, mode))}>开始训练</button>}
          {running && <button onClick={() => void runRequest(pauseSession)}>暂停</button>}
          {paused && <button className="primary" onClick={() => void runRequest(resumeSession)}>继续</button>}
          {(running || paused) && <button className="danger" onClick={() => void runRequest(stopSession)}>停止</button>}
        </div>
      </section>

      <section className="metric-grid">
        <Metric label="任务进度" value={`${progressPercent}%`} detail={`${snapshot.elapsed_s.toFixed(1)} / ${snapshot.duration_s.toFixed(1)} s`} progress={snapshot.task_progress} />
        <Metric label="当前得分" value={snapshot.score.toFixed(2)} detail={taskLabels[snapshot.task]} />
        <Metric label="交互力" value={`${Math.hypot(telemetry?.interaction_force[0] ?? 0, telemetry?.interaction_force[1] ?? 0).toFixed(3)} N`} detail="Fx / Fy / Tz" />
        <Metric label="患者主动功率" value={`${(telemetry?.human_power_w ?? 0).toFixed(3)} W`} detail={`疲劳 ${(telemetry?.fatigue ?? 0).toFixed(0)}%`} />
      </section>

      <section className="dashboard-grid">
        <Panel title="实时轨迹" subtitle="任务空间 [x, y]"><TrajectoryChart points={history} /></Panel>
        <Panel title="交互力曲线" subtitle="Fx / Fy / Tz"><LineChart values={forceSeries} colors={["#2e7dce", "#eb8a44", "#8a63d2"]} labels={["Fx", "Fy", "Tz"]} /></Panel>
        <Panel title="导纳参数" subtitle="Dx / Dy / Dθ / Ka / λv"><LineChart values={parameterSeries} colors={["#2e7dce", "#36a269", "#eb8a44", "#8a63d2", "#c75c85"]} labels={["Dx", "Dy", "Dθ", "Ka", "λv"]} /></Panel>
        <Panel title="当前状态" subtitle="20 Hz WebSocket telemetry">
          <div className="state-list">
            <StateRow label="控制模式" value={snapshot.mode === "rl" ? "RL 参数调节" : "固定导纳"} />
            <StateRow label="安全状态" value={telemetry?.safety_status ?? "idle"} tone={telemetry?.safety_status === "fallback" ? "warning" : "ok"} />
            <StateRow label="当前位置" value={telemetry ? `[${telemetry.actual_pose.map((value) => value.toFixed(3)).join(", ")}]` : "等待数据"} />
            <StateRow label="当前动作" value={telemetry ? `[${telemetry.rl_action.map((value) => value.toFixed(2)).join(", ")}]` : "[0, 0, 0, 0]"} />
          </div>
        </Panel>
      </section>

      <section className="card report-card">
        <div className="panel-heading"><div><h2>训练摘要</h2><p>会话结束后生成的可追溯指标</p></div><span className="session-id">{snapshot.session_id}</span></div>
        {snapshot.report ? <div className="report-grid">
          <ReportItem label="完成率" value={`${(snapshot.report.completion_rate * 100).toFixed(0)}%`} />
          <ReportItem label="平均轨迹误差" value={snapshot.report.average_tracking_error.toFixed(4)} />
          <ReportItem label="峰值交互力" value={`${snapshot.report.peak_interaction_force.toFixed(4)} N`} />
          <ReportItem label="运动平滑度" value={snapshot.report.motion_smoothness.toFixed(4)} />
          <ReportItem label="患者主动做功" value={`${snapshot.report.patient_active_work.toFixed(4)} J`} />
          <ReportItem label="机器人辅助做功" value={`${snapshot.report.robot_assistance_work.toFixed(4)} J`} />
        </div> : <div className="empty-report">完成一次训练后，这里会显示训练摘要。</div>}
      </section>
      <footer>Phase 7 simulation interface · RL does not publish motor or joint commands · {config?.hardware_validation_required ? "hardware validation required" : "simulation"}</footer>
    </main>
  );
}

function Metric({ label, value, detail, progress }: { label: string; value: string; detail: string; progress?: number }) {
  return <div className="metric-card card"><span>{label}</span><strong>{value}</strong><small>{detail}</small>{progress !== undefined && <div className="progress-track"><div style={{ width: `${progress * 100}%` }} /></div>}</div>;
}

function Panel({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return <div className="card panel"><div className="panel-heading"><div><h2>{title}</h2><p>{subtitle}</p></div></div>{children}</div>;
}

function StateRow({ label, value, tone = "normal" }: { label: string; value: string; tone?: "normal" | "ok" | "warning" }) {
  return <div className="state-row"><span>{label}</span><strong className={`tone-${tone}`}>{value}</strong></div>;
}

function ReportItem({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

export default App;
