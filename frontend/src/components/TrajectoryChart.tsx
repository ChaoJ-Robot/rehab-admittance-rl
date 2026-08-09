import type { Telemetry } from "../types";

interface TrajectoryChartProps {
  points: Telemetry[];
  targets?: number[][];
}

export function TrajectoryChart({ points, targets = [] }: TrajectoryChartProps) {
  const width = 620;
  const height = 260;
  const coordinatePoints = [
    ...points.flatMap((point) => [
      [point.reference_pose[0], point.reference_pose[1]],
      [point.actual_pose[0], point.actual_pose[1]]
    ]),
    ...targets.map(([x, y]) => [x, y])
  ];
  const xs = coordinatePoints.map(([x]) => x as number);
  const ys = coordinatePoints.map(([, y]) => y as number);
  const minX = xs.length ? Math.min(...xs) : 0;
  const maxX = xs.length ? Math.max(...xs) : 1;
  const minY = ys.length ? Math.min(...ys) : -0.1;
  const maxY = ys.length ? Math.max(...ys) : 0.1;
  const scaleX = maxX - minX || 1;
  const scaleY = maxY - minY || 1;
  const project = (x: number, y: number) => [
    ((x - minX) / scaleX) * (width - 24) + 12,
    height - (((y - minY) / scaleY) * (height - 24) + 12)
  ];
  const path = (key: "reference_pose" | "actual_pose") =>
    points.map((point) => project(point[key][0], point[key][1]).join(",")).join(" ");

  return (
    <div className="chart-wrap">
      <svg className="trajectory-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="trajectory chart">
        <rect x="0" y="0" width={width} height={height} rx="12" className="chart-background" />
        {[0.25, 0.5, 0.75].map((ratio) => <g key={ratio}>
          <line x1={width * ratio} x2={width * ratio} y1="0" y2={height} className="trajectory-grid" />
          <line x1="0" x2={width} y1={height * ratio} y2={height * ratio} className="trajectory-grid" />
        </g>)}
        {targets.map((target, index) => {
          const [cx, cy] = project(target[0], target[1]);
          return (
            <g key={index}>
              <circle cx={cx} cy={cy} r="9" className="task-target-ring" />
              <text x={cx} y={cy + 3} textAnchor="middle" className="task-target-label">{index + 1}</text>
            </g>
          );
        })}
        <polyline points={path("reference_pose")} className="reference-path" />
        <polyline points={path("actual_pose")} className="actual-path" />
        {points.length > 0 && <circle cx={project(points[points.length - 1].actual_pose[0], points[points.length - 1].actual_pose[1])[0]} cy={project(points[points.length - 1].actual_pose[0], points[points.length - 1].actual_pose[1])[1]} r="5" className="trajectory-endpoint" />}
        {points.length === 0 && <g className="chart-empty"><circle cx={width / 2} cy={height / 2 - 8} r="18" /><path d={`M ${width / 2 - 6} ${height / 2 - 8} h 12 M ${width / 2} ${height / 2 - 14} v 12`} /><text x={width / 2} y={height / 2 + 28} textAnchor="middle">WAITING FOR TRAJECTORY</text></g>}
      </svg>
      <div className="chart-legend">
        <span><i className="legend-reference" />参考轨迹</span>
        <span><i className="legend-actual" />实际轨迹</span>
        {targets.length > 0 && <span><i className="legend-target" />到达目标点</span>}
      </div>
    </div>
  );
}
