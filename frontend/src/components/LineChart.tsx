import type { CSSProperties } from "react";

interface LineChartProps {
  values: number[][];
  colors: string[];
  height?: number;
  labels?: string[];
}

export function LineChart({ values, colors, height = 150, labels = [] }: LineChartProps) {
  const width = 620;
  const flat = values.flat();
  const min = flat.length ? Math.min(...flat) : 0;
  const max = flat.length ? Math.max(...flat) : 1;
  const range = max - min || 1;
  const polylines = values.map((series, index) => {
    const points = series
      .map((value, pointIndex) => {
        const x = series.length <= 1 ? 0 : (pointIndex / (series.length - 1)) * width;
        const y = height - ((value - min) / range) * (height - 12) - 6;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    return <polyline key={index} points={points} fill="none" stroke={colors[index]} strokeWidth="2" />;
  });

  const legendStyle: CSSProperties = { color: "#526174" };
  return (
    <div className="chart-wrap">
      <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="telemetry chart">
        <line x1="0" x2={width} y1={height / 2} y2={height / 2} className="grid-line" />
        <line x1="0" x2={width} y1={height - 1} y2={height - 1} className="grid-line" />
        {polylines}
      </svg>
      <div className="chart-legend">
        {labels.map((label, index) => (
          <span key={label} style={legendStyle}>
            <i style={{ backgroundColor: colors[index] }} /> {label}
          </span>
        ))}
        <small>{max.toFixed(3)} / {min.toFixed(3)}</small>
      </div>
    </div>
  );
}
