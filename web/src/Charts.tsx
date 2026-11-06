import { useEffect, useRef } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  Area,
  AreaChart,
} from "recharts";
import type { Signal, Spectral } from "./types";
export const COLORS = ["#64b7a2", "#8faab2", "#d59a70"];
export function SignalChart({
  signal,
  index = 0,
  domain,
  onMinute,
}: {
  signal: Signal;
  index?: number;
  domain?: [number, number];
  onMinute?: (n: number) => void;
}) {
  return (
    <div className="signal-track">
      <div className="track-name">
        <span className="track-index">0{index + 1}</span>
        <div>
          <strong>{signal.name}</strong>
          <small>
            {index === 0
              ? "Normalized motion"
              : index === 1
                ? "0.1–0.5 Hz · motion-derived"
                : "10–15.4 Hz · EMD motion proxy"}
          </small>
        </div>
        <i style={{ background: COLORS[index] }} />
      </div>
      {signal.available ? (
        <ResponsiveContainer width="100%" height={130}>
          <LineChart
            syncId="signal-tracks"
            syncMethod="value"
            data={signal.points}
            onClick={(e) => {
              if (onMinute && e?.activeLabel !== undefined)
                onMinute(Number(e.activeLabel));
            }}
            margin={{ top: 8, right: 12, bottom: 0, left: -25 }}
          >
            <CartesianGrid vertical={false} stroke="#28372e" />
            <XAxis
              dataKey="minute"
              type="number"
              domain={domain || ["dataMin", "dataMax"]}
              allowDataOverflow
              tickFormatter={(v) => `${Number(v).toFixed(1)}m`}
              tickLine={false}
              axisLine={false}
              minTickGap={65}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={50}
              tickCount={3}
              tickFormatter={(v) => Number(v).toFixed(1)}
            />
            <Tooltip
              labelFormatter={(v) => `Minute ${Number(v).toFixed(2)}`}
              formatter={(v) => Number(v).toFixed(4)}
              contentStyle={{ borderRadius: 8, border: "1px solid #3b5041" }}
            />
            <Line
              type="linear"
              dataKey="value"
              stroke={COLORS[index]}
              strokeWidth={1.2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="empty-track">
          No component met the extraction rule in this segment.
        </div>
      )}
    </div>
  );
}
export function Curve({
  data,
  title,
  xLabel,
  yLabel,
  color = "#64b7a2",
  reference = false,
}: {
  data: { x: number; y: number }[];
  title: string;
  xLabel: string;
  yLabel: string;
  color?: string;
  reference?: boolean;
}) {
  return (
    <div className="chart-panel">
      <h3>{title}</h3>
      <span className="eyebrow">{yLabel}</span>
      <ResponsiveContainer width="100%" height={235}>
        <AreaChart
          data={data}
          margin={{ top: 14, right: 18, bottom: 5, left: -15 }}
        >
          <CartesianGrid stroke="#28372e" strokeDasharray="3 4" />
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, 1]}
            tickLine={false}
            tickCount={5}
          />
          <YAxis domain={[0, 1]} tickCount={5} tickLine={false} />
          <Tooltip
            formatter={(v) => Number(v).toFixed(3)}
            labelFormatter={(v) => `${xLabel}: ${Number(v).toFixed(3)}`}
          />
          {reference ? (
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              stroke="#718a77"
              strokeDasharray="5 5"
            />
          ) : null}
          <Area
            dataKey="y"
            stroke={color}
            fill={color}
            fillOpacity={0.08}
            strokeWidth={2}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
      <small className="axis-label">{xLabel}</small>
    </div>
  );
}
export function Spectrogram({ data }: { data: Spectral }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const w = data.minutes.length,
      h = data.frequency.length;
    c.width = w;
    c.height = h;
    const flat = data.power.flat();
    const lo = Math.min(...flat),
      hi = Math.max(...flat);
    const image = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const v = (data.power[h - y - 1][x] - lo) / (hi - lo || 1);
        const i = (y * w + x) * 4;
        image.data[i] = 15 + v * 205;
        image.data[i + 1] = 48 + v * 148;
        image.data[i + 2] = 52 + v * 63;
        image.data[i + 3] = 255;
      }
    ctx.putImageData(image, 0, 0);
  }, [data]);
  return (
    <div className="spectral">
      <div>
        <strong>Respiratory spectrogram</strong>
        <small>
          Relative power · dB · {data.frequency[0]?.toFixed(2)}–
          {data.frequency.at(-1)?.toFixed(2)} Hz
        </small>
      </div>
      <canvas
        ref={ref}
        role="img"
        aria-label="Respiratory spectral power across the selected segment"
      />
      <div className="spectrum-scale">
        <span>Lower power</span>
        <i />
        <span>Higher power</span>
      </div>
    </div>
  );
}
