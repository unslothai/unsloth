// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The results chart in three shapes over the same runs:
//   bars   one bar per setting, mean with a min–max whisker
//   runs   each setting's runs in order, which is where a warming cache shows
//   depth  throughput against draft depth, one line per speculative family
// The SVG carries literal colours, so Download SVG / PNG saves what is on screen.

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
import { BarRows, type BarRow, barRowsHeight } from "./bar-chart";
import {
  type Tip,
  ChartFrame,
  chartHead,
  chartHeight,
  footerLines,
  useInk,
} from "./chart-frame";
import { AXIS_H, RIGHT_W, TOP, W, clip, niceMax, tickText, ticks } from "./chart-math";
import { useFamilyColors } from "./family-colors";

export type ChartKind = "bars" | "runs" | "depth";

const LINE_H = 320;
const LEFT_AXIS = 70;

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

export interface PendingRow {
  label: string;
  family: Family;
  /** "Queued", "Loading", "Measuring 2/3" */
  status: string;
  active: boolean;
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
  bare = false,
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
  /** On screen: no title, subtitle or fine print. The page around it already says them. */
  bare?: boolean;
}): ReactElement {
  const palette = useFamilyColors();
  const { ink, muted, grid, card } = useInk();
  const [tip, setTip] = useState<Tip | null>(null);
  const baseline = rows.find((r) => r.isBaseline) ?? null;
  const head = chartHead(bare, subtitle);

  let plotH: number;
  let body: ReactElement;
  const points: Tip[] = [];

  if (kind === "bars") {
    const groups = [
      {
        rows: rows.map(
          (r): BarRow => ({
            label: r.label,
            value: r.mean,
            low: r.min,
            high: r.max,
            color: palette[r.family],
            quiet: r.isBaseline,
            tail: r.isBaseline ? "baseline" : fmtPct(r.pct),
            tip: [
              r.label,
              `${fmtRate(r.mean)}${r.n > 1 ? ` · ${r.min.toFixed(1)}–${r.max.toFixed(1)} over ${r.n} runs` : " · 1 run"}`,
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
            ].filter(Boolean),
          }),
        ),
      },
    ];
    const queued = pending.map((p) => ({ ...p, color: palette[p.family] }));
    plotH = barRowsHeight(groups, queued);
    body = (
      <BarRows
        groups={groups}
        pending={queued}
        max={niceMax(Math.max(10, ...rows.map((r) => r.max)))}
        format={fmtRate}
        axisLabel="Generation throughput (tokens / sec)"
        marker={baseline?.mean ?? null}
        ink={{ ink, muted, grid, card }}
        onTip={(t) => setTip(t && { ...t, y: head + t.y })}
      />
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

  const footLines = footerLines(bare, footer);
  const height = chartHeight(head, plotH, footLines);
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
    <ChartFrame
      title={title}
      subtitle={subtitle}
      footLines={footLines}
      bare={bare}
      head={head}
      plotH={plotH}
      height={height}
      card={card}
      ink={ink}
      muted={muted}
      svgRef={svgRef}
      tip={tip}
      onMouseMove={onMove}
      onMouseLeave={() => setTip(null)}
    >
      {body}
    </ChartFrame>
  );
});
