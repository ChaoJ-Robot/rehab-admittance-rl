import type { Telemetry } from "../types";

interface TrajectoryChartProps {
  points: Telemetry[];
}

export function TrajectoryChart({ points }: TrajectoryChartProps) {
  const width = 620;
  const height = 260;
  const coordinates = points.flatMap((point) => [
    [point.reference_pose[0], point.reference_pose[1]],
    [point.actual_pose[0], point.actual_pose[1]]
  ]);
  const xs = coordinates.map(([x]) => x);
  const ys = coordinates.map(([, y]) => y);
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
        <polyline points={path("reference_pose")} className="reference-path" />
        <polyline points={path("actual_pose")} className="actual-path" />
      </svg>
      <div className="chart-legend">
        <span><i className="legend-reference" />参考轨迹</span>
        <span><i className="legend-actual" />实际轨迹</span>
      </div>
    </div>
  );
}
