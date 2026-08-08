import type { Telemetry } from "../types";

interface MazeChartProps {
  points: Telemetry[];
}

/** Fixed world bounds of the maze workspace used for projection. */
const BOUNDS = { minX: 0.12, maxX: 0.63, minY: -0.26, maxY: 0.26 };
const WIDTH = 620;
const HEIGHT = 300;

function project(x: number, y: number): [number, number] {
  const px = ((x - BOUNDS.minX) / (BOUNDS.maxX - BOUNDS.minX)) * (WIDTH - 24) + 12;
  const py = HEIGHT - (((y - BOUNDS.minY) / (BOUNDS.maxY - BOUNDS.minY)) * (HEIGHT - 24) + 12);
  return [px, py];
}

export function MazeChart({ points }: MazeChartProps) {
  const latest = points.length > 0 ? points[points.length - 1] : null;
  const walls = latest?.maze_walls ?? [];
  const actualPath = points
    .map((point) => project(point.actual_pose[0], point.actual_pose[1]).join(","))
    .join(" ");
  const referencePath = points
    .map((point) => project(point.reference_pose[0], point.reference_pose[1]).join(","))
    .join(" ");
  const endpoint = latest
    ? project(latest.actual_pose[0], latest.actual_pose[1])
    : null;

  return (
    <div className="chart-wrap">
      <svg className="trajectory-chart" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="maze chart">
        <rect x="0" y="0" width={WIDTH} height={HEIGHT} rx="12" className="chart-background" />
        {[0.25, 0.5, 0.75].map((ratio) => (
          <g key={ratio}>
            <line x1={WIDTH * ratio} x2={WIDTH * ratio} y1="0" y2={HEIGHT} className="trajectory-grid" />
            <line x1="0" x2={WIDTH} y1={HEIGHT * ratio} y2={HEIGHT * ratio} className="trajectory-grid" />
          </g>
        ))}
        {walls.map((wall, index) => {
          const [x1, y1] = project(wall[0], wall[1]);
          const [x2, y2] = project(wall[2], wall[3]);
          return <line key={index} x1={x1} y1={y1} x2={x2} y2={y2} className="maze-wall" />;
        })}
        <polyline points={referencePath} className="reference-path" />
        <polyline points={actualPath} className="actual-path" />
        {endpoint && (
          <circle cx={endpoint[0]} cy={endpoint[1]} r="5" className="trajectory-endpoint" />
        )}
        {!latest && (
          <g className="chart-empty">
            <circle cx={WIDTH / 2} cy={HEIGHT / 2 - 8} r="18" />
            <text x={WIDTH / 2} y={HEIGHT / 2 + 28} textAnchor="middle">WAITING FOR TELEMETRY</text>
          </g>
        )}
      </svg>
      <div className="chart-legend">
        <span><i className="legend-wall" />迷宫墙</span>
        <span><i className="legend-reference" />参考路径</span>
        <span><i className="legend-actual" />实际轨迹</span>
      </div>
    </div>
  );
}
