import type { Telemetry } from "../types";

interface ColorMemoryPanelProps {
  points: Telemetry[];
}

const COLOR_HEX: Record<string, string> = {
  red: "#ff6b6b",
  blue: "#4dabf7",
  green: "#51cf66",
  yellow: "#ffd43b"
};
const COLOR_LABEL: Record<string, string> = {
  red: "红",
  blue: "蓝",
  green: "绿",
  yellow: "黄"
};

const WIDTH = 620;
const HEIGHT = 300;
const BOUNDS = { minX: 0.18, maxX: 0.54, minY: -0.2, maxY: 0.2 };

function project(x: number, y: number): [number, number] {
  const px = ((x - BOUNDS.minX) / (BOUNDS.maxX - BOUNDS.minX)) * (WIDTH - 80) + 40;
  const py = HEIGHT - (((y - BOUNDS.minY) / (BOUNDS.maxY - BOUNDS.minY)) * (HEIGHT - 100) + 50);
  return [px, py];
}

export function ColorMemoryPanel({ points }: ColorMemoryPanelProps) {
  const latest = points.length > 0 ? points[points.length - 1] : null;
  const positions = latest?.color_block_positions ?? [];
  const names = latest?.color_block_names ?? [];
  const sequence = latest?.color_sequence ?? [];
  const phase = latest?.task_phase ?? null;
  // Current recall target = the sequenced colour block nearest to the reference pose.
  let currentIndex = -1;
  if (phase === "recall" && latest && positions.length > 0) {
    let bestDistance = Infinity;
    sequence.forEach((color, sequenceIndex) => {
      const blockIndex = names.indexOf(color);
      if (blockIndex < 0) return;
      const block = positions[blockIndex];
      const distance = Math.hypot(
        block[0] - latest.reference_pose[0],
        block[1] - latest.reference_pose[1]
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        currentIndex = sequenceIndex;
      }
    });
  }
  const endpoint = latest ? project(latest.actual_pose[0], latest.actual_pose[1]) : null;

  return (
    <div className="chart-wrap">
      <div className="memory-sequence-bar">
        <span className={`memory-phase phase-${phase ?? "idle"}`}>
          {phase === "memorize" ? "记忆阶段" : phase === "recall" ? "复述阶段" : "待开始"}
        </span>
        <div className="memory-sequence">
          {sequence.map((color, index) => (
            <span
              key={index}
              className={`memory-step ${phase === "recall" && index === currentIndex ? "current" : ""}`}
              style={{ background: COLOR_HEX[color] ?? "#888" }}
            >
              {index + 1}
            </span>
          ))}
        </div>
      </div>
      <svg className="trajectory-chart" viewBox={`0 0 ${WIDTH} ${HEIGHT - 60}`} role="img" aria-label="color memory chart">
        <rect x="0" y="0" width={WIDTH} height={HEIGHT - 60} rx="12" className="chart-background" />
        {positions.map((position, index) => {
          const [cx, cy] = project(position[0], position[1]);
          const name = names[index] ?? "";
          return (
            <g key={index}>
              <rect
                x={cx - 26}
                y={cy - 26}
                width="52"
                height="52"
                rx="10"
                fill={COLOR_HEX[name] ?? "#888"}
                opacity="0.85"
                className="color-block"
              />
              <text x={cx} y={cy + 5} textAnchor="middle" className="color-block-label">
                {COLOR_LABEL[name] ?? name}
              </text>
            </g>
          );
        })}
        {endpoint && (
          <circle cx={endpoint[0]} cy={endpoint[1]} r="6" className="trajectory-endpoint" />
        )}
        {!latest && (
          <g className="chart-empty">
            <circle cx={WIDTH / 2} cy={110} r="18" />
            <text x={WIDTH / 2} y={148} textAnchor="middle">WAITING FOR TELEMETRY</text>
          </g>
        )}
      </svg>
      <div className="chart-legend">
        <span><i className="legend-actual" />末端位置</span>
        <span className="memory-hint">记忆序列后，按顺序将末端移动到对应色块</span>
      </div>
    </div>
  );
}
