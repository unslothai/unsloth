// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The results chart in three shapes over the same runs:
//   bars   one bar per setting, mean with a min–max whisker
//   runs   each setting's runs in order, which is where a warming cache shows
//   depth  throughput against draft depth, one line per speculative family
// The SVG carries literal colours, so Download SVG / PNG saves what is on screen.

import { cn } from "@/lib/utils";
import { useTheme } from "@/features/settings";
import { type ReactElement, type RefObject, memo, useState } from "react";
import {
  type AggRow,
  type DepthSeries,
  type Family,
  FAMILY_LABEL,
  type RunSeries,
  fmtMs,
  fmtPct,
  fmtRate,
} from "../lib/bench-math";
import { useFamilyColors } from "./family-colors";

export type ChartKind = "bars" | "runs" | "depth";

let probe: CanvasRenderingContext2D | null | undefined;

/** Any CSS colour folded to #rrggbb, since many SVG viewers can't paint oklch(). */
function toPortableColor(value: string): string {
  if (typeof document === "undefined") return value;
  if (probe === undefined) {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    probe = canvas.getContext("2d", { willReadFrequently: true });
  }
  if (!probe) return value;
  probe.clearRect(0, 0, 1, 1);
  probe.fillStyle = "#000";
  probe.fillStyle = value;
  probe.fillRect(0, 0, 1, 1);
  const [r, g, b, a] = probe.getImageData(0, 0, 1, 1).data;
  const hex = (n: number) => n.toString(16).padStart(2, "0");
  return a === 255
    ? `#${hex(r)}${hex(g)}${hex(b)}`
    : `rgba(${r}, ${g}, ${b}, ${(a / 255).toFixed(3)})`;
}

function useInk(): { ink: string; muted: string; grid: string; card: string } {
  // Subscribing re-renders on a theme flip, which is when the tokens change.
  useTheme();
  const style = getComputedStyle(document.documentElement);
  const read = (n: string, fallback: string) =>
    toPortableColor(style.getPropertyValue(n).trim() || fallback);
  return {
    ink: read("--foreground", "#111"),
    muted: read("--muted-foreground", "#777"),
    grid: read("--border", "#ddd"),
    card: read("--card", "#fff"),
  };
}

const W = 1000;
const LABEL_W = 300;
const RIGHT_W = 170;
const ROW_H_FULL = 46;
const BAR_H_FULL = 26;
const TOP = 18;
const AXIS_H = 52;
const LINE_H = 320;
const LEFT_AXIS = 70;
const FOOT_LINE = 17;
const FONT = "Inter Variable, ui-sans-serif, system-ui, sans-serif";

/** SVG text never wraps: break long footer lines on their " · " joins, about 160 chars at 11px. */
const FOOT_CHARS = 160;
function wrapFooter(lines: string[]): string[] {
  const out: string[] = [];
  for (const line of lines) {
    let cur = "";
    for (const part of line.split(" · ")) {
      const next = cur ? `${cur} · ${part}` : part;
      if (cur && next.length > FOOT_CHARS) {
        out.push(cur);
        cur = part;
      } else cur = next;
    }
    if (cur) out.push(cur);
  }
  return out;
}

function niceMax(max: number): number {
  if (max <= 0) return 1;
  const base = 10 ** Math.floor(Math.log10(max));
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10])
    if (m * base >= max * 1.08) return m * base;
  return 10 * base;
}

function ticks(max: number): number[] {
  return Array.from({ length: 6 }, (_, i) =>
    Number(((i * max) / 5).toFixed(6)),
  );
}

function tickText(t: number): string {
  return Number.isInteger(t) ? String(t) : t.toFixed(1);
}

/** A bar anchored flat at the baseline with only its data end rounded. */
function barPath(
  x0: number,
  x1: number,
  y: number,
  h: number,
  r: number,
): string {
  const rr = Math.min(r, Math.max(0, x1 - x0), h / 2);
  return `M${x0},${y} H${x1 - rr} a${rr},${rr} 0 0 1 ${rr},${rr} V${y + h - rr} a${rr},${rr} 0 0 1 -${rr},${rr} H${x0} Z`;
}

/** Push overlapping end labels apart, keeping their order. */
function spreadLabels(
  ys: number[],
  gap: number,
  lo: number,
  hi: number,
): number[] {
  const idx = ys.map((_, i) => i).sort((a, b) => ys[a] - ys[b]);
  const out = ys.slice();
  let prev = Number.NEGATIVE_INFINITY;
  for (const i of idx) {
    out[i] = Math.max(ys[i], prev + gap, lo);
    prev = out[i];
  }
  const over = out.length ? Math.max(...out) - hi : 0;
  if (over > 0) for (const i of idx) out[i] -= over;
  return out;
}

function clip(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

export interface PendingRow {
  label: string;
  family: Family;
  /** "Queued", "Loading", "Measuring 2/3" */
  status: string;
  active: boolean;
}

interface Tip {
  x: number;
  y: number;
  lines: string[];
}

export const BenchChart = memo(function BenchChart({
  kind,
  rows,
  series,
  depth,
  title,
  subtitle,
  footer,
  svgRef,
  pending = [],
}: {
  kind: ChartKind;
  rows: AggRow[];
  series: RunSeries[];
  depth: DepthSeries[];
  title: string;
  subtitle?: string;
  footer: string[];
  svgRef?: RefObject<SVGSVGElement | null>;
  /** Rows still to come in a live run, drawn as placeholders under the measured ones. */
  pending?: PendingRow[];
}): ReactElement {
  const palette = useFamilyColors();
  const { ink, muted, grid, card } = useInk();
  const [tip, setTip] = useState<Tip | null>(null);
  const baseline = rows.find((r) => r.isBaseline) ?? null;
  const head = subtitle ? 70 : 48;

  let plotH: number;
  let body: ReactElement;
  const points: Tip[] = [];

  if (kind === "bars") {
    const count = rows.length + pending.length;
    const ROW_H = count > 14 ? 28 : count > 9 ? 34 : ROW_H_FULL;
    const BAR_H = count > 14 ? 17 : count > 9 ? 21 : BAR_H_FULL;
    const FS = count > 14 ? 12.5 : 13.5;
    plotH = TOP + count * ROW_H + AXIS_H;
    const maxVal = niceMax(Math.max(10, ...rows.map((r) => r.max)));
    const plotW = W - LABEL_W - RIGHT_W;
    const x = (v: number) => LABEL_W + (v / maxVal) * plotW;
    const bottom = TOP + count * ROW_H;
    body = (
      <g>
        {ticks(maxVal).map((t) => (
          <g key={t}>
            <line
              x1={x(t)}
              x2={x(t)}
              y1={TOP - 6}
              y2={bottom}
              stroke={grid}
              strokeWidth={1}
            />
            <text
              x={x(t)}
              y={bottom + 20}
              fontSize={12}
              textAnchor="middle"
              fill={muted}
            >
              {tickText(t)}
            </text>
          </g>
        ))}
        <text
          x={LABEL_W + plotW / 2}
          y={bottom + 42}
          fontSize={12.5}
          textAnchor="middle"
          fill={muted}
        >
          Generation throughput (tokens / sec)
        </text>
        {rows.map((r, i) => {
          const y = TOP + i * ROW_H + (ROW_H - BAR_H) / 2;
          const cy = y + BAR_H / 2;
          const value = fmtRate(r.mean);
          const tail = r.isBaseline ? "baseline" : fmtPct(r.pct);
          const lines = [
            r.label,
            `${value}${r.n > 1 ? ` · ${r.min.toFixed(1)}–${r.max.toFixed(1)} over ${r.n} runs` : " · 1 run"}`,
            [
              r.ttftMs !== null ? `first token ${fmtMs(r.ttftMs)}` : "",
              r.loadMs !== null ? `load ${fmtMs(r.loadMs)}` : "",
            ]
              .filter(Boolean)
              .join(" · "),
            r.acceptRate !== null
              ? `${Math.round(r.acceptRate * 100)}% of drafted tokens accepted`
              : "",
            r.clientOnly ? "No server timings; timed in the browser" : "",
          ].filter(Boolean);
          return (
            <g
              key={r.label}
              onMouseEnter={() => setTip({ x: x(r.mean), y: head + cy, lines })}
              onMouseLeave={() => setTip(null)}
            >
              <rect
                x={0}
                y={TOP + i * ROW_H}
                width={W}
                height={ROW_H}
                fill="transparent"
              />
              <text
                x={LABEL_W - 14}
                y={cy + 4.5}
                fontSize={FS}
                textAnchor="end"
                fill={r.isBaseline ? muted : ink}
              >
                {clip(r.label, 40)}
              </text>
              <path
                d={barPath(
                  LABEL_W,
                  Math.max(LABEL_W + 2, x(r.mean)),
                  y,
                  BAR_H,
                  4,
                )}
                fill={palette[r.family]}
              />
              {r.n > 1 && r.max > r.min && (
                <g stroke={ink} strokeOpacity={0.55} strokeWidth={1.5}>
                  <line x1={x(r.min)} x2={x(r.max)} y1={cy} y2={cy} />
                  <line x1={x(r.min)} x2={x(r.min)} y1={cy - 4} y2={cy + 4} />
                  <line x1={x(r.max)} x2={x(r.max)} y1={cy - 4} y2={cy + 4} />
                </g>
              )}
              <text x={x(r.max) + 12} y={cy + 4.5} fontSize={FS} fill={ink}>
                <tspan fontWeight={650}>{value}</tspan>
                <tspan dx={8} fill={muted}>
                  {tail}
                </tspan>
              </text>
            </g>
          );
        })}
        {pending.map((p, j) => {
          const i = rows.length + j;
          const y = TOP + i * ROW_H + (ROW_H - BAR_H) / 2;
          const cy = y + BAR_H / 2;
          const color = palette[p.family];
          return (
            <g key={`pending-${p.label}`}>
              <text
                x={LABEL_W - 14}
                y={cy + 4.5}
                fontSize={FS}
                textAnchor="end"
                fill={p.active ? ink : muted}
              >
                {clip(p.label, 40)}
              </text>
              <rect
                x={LABEL_W}
                y={y}
                width={plotW}
                height={BAR_H}
                rx={4}
                fill={p.active ? color : grid}
                fillOpacity={p.active ? 0.16 : 0.35}
              />
              {p.active && (
                <rect x={LABEL_W} y={y} width={4} height={BAR_H} rx={2} fill={color} />
              )}
              <text
                x={LABEL_W + 14}
                y={cy + 4.5}
                fontSize={12}
                fill={p.active ? ink : muted}
              >
                {p.status}
              </text>
            </g>
          );
        })}
        {baseline && (
          <g>
            <line
              x1={x(baseline.mean)}
              x2={x(baseline.mean)}
              y1={TOP - 8}
              y2={bottom}
              stroke={muted}
              strokeWidth={1.5}
              strokeDasharray="5 4"
            />
          </g>
        )}
      </g>
    );
  } else if (kind === "runs") {
    plotH = TOP + LINE_H + AXIS_H;
    const maxSeq = Math.max(
      2,
      ...series.flatMap((s) => s.points.map((p) => p.seq)),
    );
    const maxVal = niceMax(
      Math.max(0, ...series.flatMap((s) => s.points.map((p) => p.value))),
    );
    const plotW = W - LEFT_AXIS - RIGHT_W - 60;
    const x = (seq: number) => LEFT_AXIS + ((seq - 1) / (maxSeq - 1)) * plotW;
    const y = (v: number) => TOP + LINE_H - (v / maxVal) * LINE_H;
    const endYs = spreadLabels(
      series.map((s) => y(s.points[s.points.length - 1].value)),
      15,
      TOP,
      TOP + LINE_H,
    );
    for (const s of series)
      for (const p of s.points)
        points.push({
          x: x(p.seq),
          y: head + y(p.value),
          lines: [
            s.label,
            `run ${p.seq}${p.warmup ? " (warm-up)" : ""}: ${fmtRate(p.value)}`,
          ],
        });
    body = (
      <g>
        {ticks(maxVal).map((t) => (
          <g key={t}>
            <line
              x1={LEFT_AXIS}
              x2={LEFT_AXIS + plotW}
              y1={y(t)}
              y2={y(t)}
              stroke={grid}
              strokeWidth={1}
            />
            <text
              x={LEFT_AXIS - 10}
              y={y(t) + 4}
              fontSize={12}
              textAnchor="end"
              fill={muted}
            >
              {tickText(t)}
            </text>
          </g>
        ))}
        {Array.from({ length: maxSeq }, (_, i) => i + 1).map((seq) => (
          <text
            key={seq}
            x={x(seq)}
            y={TOP + LINE_H + 20}
            fontSize={12}
            textAnchor="middle"
            fill={muted}
          >
            {seq}
          </text>
        ))}
        <text
          x={LEFT_AXIS + plotW / 2}
          y={TOP + LINE_H + 42}
          fontSize={12.5}
          textAnchor="middle"
          fill={muted}
        >
          Run, in order (hollow = warm-up)
        </text>
        {series.map((s, si) => (
          <g key={s.label}>
            <path
              d={s.points
                .map(
                  (p, i) => `${i === 0 ? "M" : "L"}${x(p.seq)},${y(p.value)}`,
                )
                .join(" ")}
              fill="none"
              stroke={palette[s.family]}
              strokeWidth={2}
              strokeLinejoin="round"
            />
            {s.points.map((p) => (
              <circle
                key={p.seq}
                cx={x(p.seq)}
                cy={y(p.value)}
                r={4.5}
                fill={p.warmup ? card : palette[s.family]}
                stroke={p.warmup ? palette[s.family] : card}
                strokeWidth={2}
              />
            ))}
            <text
              x={x(s.points[s.points.length - 1].seq) + 10}
              y={endYs[si] + 4}
              fontSize={11.5}
              fill={ink}
            >
              {clip(s.label, 30)}
            </text>
          </g>
        ))}
      </g>
    );
  } else {
    plotH = TOP + LINE_H + AXIS_H;
    const ns = [
      ...new Set(depth.flatMap((s) => s.points.map((p) => p.n))),
    ].sort((a, b) => a - b);
    const lo = ns[0] ?? 1;
    const hi = ns[ns.length - 1] ?? 2;
    const maxVal = niceMax(
      Math.max(
        0,
        ...depth.flatMap((s) => s.points.map((p) => p.max)),
        baseline?.mean ?? 0,
      ),
    );
    const plotW = W - LEFT_AXIS - RIGHT_W - 60;
    const x = (n: number) =>
      LEFT_AXIS + 24 + ((n - lo) / Math.max(1, hi - lo)) * (plotW - 48);
    const y = (v: number) => TOP + LINE_H - (v / maxVal) * LINE_H;
    const endYs = spreadLabels(
      depth.map((s) => y(s.points[s.points.length - 1].mean)),
      15,
      TOP,
      TOP + LINE_H,
    );
    for (const s of depth)
      for (const p of s.points)
        points.push({
          x: x(p.n),
          y: head + y(p.mean),
          lines: [
            p.label,
            `${fmtRate(p.mean)}${p.max > p.min ? ` · ${p.min.toFixed(1)}–${p.max.toFixed(1)}` : ""}`,
          ],
        });
    body = (
      <g>
        {ticks(maxVal).map((t) => (
          <g key={t}>
            <line
              x1={LEFT_AXIS}
              x2={LEFT_AXIS + plotW}
              y1={y(t)}
              y2={y(t)}
              stroke={grid}
              strokeWidth={1}
            />
            <text
              x={LEFT_AXIS - 10}
              y={y(t) + 4}
              fontSize={12}
              textAnchor="end"
              fill={muted}
            >
              {tickText(t)}
            </text>
          </g>
        ))}
        {ns.map((n) => (
          <text
            key={n}
            x={x(n)}
            y={TOP + LINE_H + 20}
            fontSize={12}
            textAnchor="middle"
            fill={muted}
          >
            {n}
          </text>
        ))}
        <text
          x={LEFT_AXIS + plotW / 2}
          y={TOP + LINE_H + 42}
          fontSize={12.5}
          textAnchor="middle"
          fill={muted}
        >
          Draft tokens per step
        </text>
        {baseline && (
          <g>
            <line
              x1={LEFT_AXIS}
              x2={LEFT_AXIS + plotW}
              y1={y(baseline.mean)}
              y2={y(baseline.mean)}
              stroke={muted}
              strokeWidth={1.5}
              strokeDasharray="5 4"
            />
            <text
              x={LEFT_AXIS + plotW + 10}
              y={y(baseline.mean) + 4}
              fontSize={11.5}
              fill={muted}
            >
              {clip(baseline.label, 18)} {fmtRate(baseline.mean)}
            </text>
          </g>
        )}
        {depth.map((s, si) => (
          <g key={s.family}>
            {s.points.map(
              (p) =>
                p.max > p.min && (
                  <line
                    key={p.n}
                    x1={x(p.n)}
                    x2={x(p.n)}
                    y1={y(p.min)}
                    y2={y(p.max)}
                    stroke={palette[s.family]}
                    strokeWidth={1.5}
                    strokeOpacity={0.55}
                  />
                ),
            )}
            <path
              d={s.points
                .map((p, i) => `${i === 0 ? "M" : "L"}${x(p.n)},${y(p.mean)}`)
                .join(" ")}
              fill="none"
              stroke={palette[s.family]}
              strokeWidth={2}
              strokeLinejoin="round"
            />
            {s.points.map((p) => (
              <circle
                key={p.n}
                cx={x(p.n)}
                cy={y(p.mean)}
                r={5}
                fill={palette[s.family]}
                stroke={card}
                strokeWidth={2}
              />
            ))}
            <text
              x={x(s.points[s.points.length - 1].n) + 12}
              y={endYs[si] + 4}
              fontSize={12}
              fill={ink}
            >
              {FAMILY_LABEL[s.family]}
            </text>
          </g>
        ))}
      </g>
    );
  }

  const footLines = wrapFooter(footer);
  const height =
    head +
    plotH +
    (footLines.length ? 10 + footLines.length * FOOT_LINE : 0) +
    10;
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (kind === "bars" || points.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const vx = ((e.clientX - rect.left) * W) / rect.width;
    const vy = ((e.clientY - rect.top) * height) / rect.height;
    let best: Tip | null = null;
    let bestD = 28;
    for (const p of points) {
      const d = Math.hypot(p.x - vx, p.y - vy);
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    setTip(best);
  };

  return (
    <div className="relative mx-auto max-w-[calc(1000px*var(--ui-space-scale,1))]">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${height}`}
        width="100%"
        className="mx-auto block max-w-[calc(1000px*var(--ui-space-scale,1))]"
        role="img"
        aria-label={title}
        style={{ fontFamily: FONT }}
        onMouseMove={onMove}
        onMouseLeave={() => setTip(null)}
      >
        <rect x={0} y={0} width={W} height={height} fill={card} />
        <text x={24} y={30} fontSize={18} fontWeight={650} fill={ink}>
          {title}
        </text>
        {subtitle && (
          <text x={24} y={52} fontSize={12.5} fill={muted}>
            {subtitle}
          </text>
        )}
        <g transform={`translate(0, ${head})`}>{body}</g>
        {footLines.map((line, i) => (
          <text
            key={`${i}-${line}`}
            x={24}
            y={head + plotH + 14 + i * FOOT_LINE}
            fontSize={11}
            fill={muted}
          >
            {line}
          </text>
        ))}
      </svg>
      {tip && (
        <div
          className="pointer-events-none absolute z-10 min-w-44 rounded-lg border border-border/60 bg-popover px-3 py-2 text-ui-12 shadow-lg"
          style={{
            left: `${(Math.min(tip.x, W - 300) / W) * 100}%`,
            top: `${(tip.y / height) * 100}%`,
            transform: "translate(14px, 10px)",
          }}
        >
          {tip.lines.map((l, i) => (
            <p
              key={l}
              className={cn(
                i === 0
                  ? "font-semibold text-foreground"
                  : "tabular-nums text-muted-foreground",
              )}
            >
              {l}
            </p>
          ))}
        </div>
      )}
    </div>
  );
});
