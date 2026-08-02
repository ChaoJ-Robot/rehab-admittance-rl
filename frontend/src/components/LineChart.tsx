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
  const hasData = flat.length > 0;
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
    const last = series[series.length - 1];
    const lastY = last === undefined ? 0 : height - ((last - min) / range) * (height - 12) - 6;
    return <g key={index}>
      <polyline points={points} fill="none" stroke={colors[index]} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      {series.length > 0 && <circle cx={width} cy={lastY} r="3.5" fill={colors[index]} className="chart-endpoint" />}
    </g>;
  });

  const legendStyle: CSSProperties = { color: "#526174" };
  return (
    <div className="chart-wrap">
      <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="telemetry chart">
        <rect x="0" y="0" width={width} height={height} rx="10" className="chart-frame" />
        {[0.2, 0.4, 0.6, 0.8].map((ratio) => <line key={ratio} x1="0" x2={width} y1={height * ratio} y2={height * ratio} className="grid-line" />)}
        {polylines}
        {!hasData && <g className="chart-empty"><circle cx={width / 2} cy={height / 2 - 8} r="15" /><path d={`M ${width / 2 - 5} ${height / 2 - 8} h 10 M ${width / 2} ${height / 2 - 13} v 10`} /><text x={width / 2} y={height / 2 + 25} textAnchor="middle">WAITING FOR TELEMETRY</text></g>}
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
