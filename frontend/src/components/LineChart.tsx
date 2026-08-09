import { type CSSProperties, useEffect, useRef, useState } from "react";

interface LineChartProps {
  values: number[][];
  colors: string[];
  height?: number;
  labels?: string[];
}

/** Minimum vertical space reserved for the chart body, in px. */
const MIN_HEIGHT = 90;
/** Chart body aspect ratio (width / height) used when the panel is wide. */
const ASPECT = 2.5;

export function LineChart({ values, colors, height, labels = [] }: LineChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(620);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const next = Math.floor(entries[0]?.contentRect.width ?? 620);
      if (next > 0) setWidth(next);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const chartHeight = Math.max(MIN_HEIGHT, Math.round(width / ASPECT));
  const flat = values.flat();
  const hasData = flat.length > 0;
  const min = flat.length ? Math.min(...flat) : 0;
  const max = flat.length ? Math.max(...flat) : 1;
  const range = max - min || 1;
  const padY = 8;
  const polylines = values.map((series, index) => {
    const points = series
      .map((value, pointIndex) => {
        const x = series.length <= 1 ? 0 : (pointIndex / (series.length - 1)) * width;
        const y = chartHeight - padY - ((value - min) / range) * (chartHeight - padY * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    const last = series[series.length - 1];
    const lastY =
      last === undefined ? 0 : chartHeight - padY - ((last - min) / range) * (chartHeight - padY * 2);
    return <g key={index}>
      <polyline points={points} fill="none" stroke={colors[index]} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      {series.length > 0 && <circle cx={width} cy={lastY} r="3.5" fill={colors[index]} className="chart-endpoint" />}
    </g>;
  });

  const legendStyle: CSSProperties = { color: "#526174" };
  return (
    <div className="chart-wrap" ref={containerRef}>
      <svg className="line-chart" viewBox={`0 0 ${width} ${chartHeight}`} preserveAspectRatio="none" role="img" aria-label="telemetry chart">
        <rect x="0" y="0" width={width} height={chartHeight} rx="10" className="chart-frame" />
        {[0.2, 0.4, 0.6, 0.8].map((ratio) => <line key={ratio} x1="0" x2={width} y1={chartHeight * ratio} y2={chartHeight * ratio} className="grid-line" />)}
        {polylines}
        {!hasData && <g className="chart-empty"><circle cx={width / 2} cy={chartHeight / 2 - 8} r="15" /><path d={`M ${width / 2 - 5} ${chartHeight / 2 - 8} h 10 M ${width / 2} ${chartHeight / 2 - 13} v 10`} /><text x={width / 2} y={chartHeight / 2 + 25} textAnchor="middle">WAITING FOR TELEMETRY</text></g>}
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
