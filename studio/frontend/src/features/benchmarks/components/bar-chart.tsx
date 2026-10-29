// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Horizontal bars on one axis: Config sweeps' throughput, Evals' scores and llama-bench's rates
// all draw through here, so the three tabs read as one page.

import {
  Fragment,
  type ReactElement,
  type RefObject,
  memo,
  useState,
} from "react";
import {
  type Ink,
  type Tip,
  ChartFrame,
  chartHead,
  chartHeight,
  footerLines,
  useInk,
} from "./chart-frame";
import {
  LABEL_W,
  RIGHT_W,
  TOP,
  W,
  barPath,
  clampWhisker,
  clip,
  layoutBars,
  tickText,
  ticks,
} from "./chart-math";

export interface BarRow {
  key?: string;
  label: string;
  value: number;
  /** Whisker ends (min–max, ± stderr); clamped to the axis, skipped when equal. */
  low?: number | null;
  high?: number | null;
  color: string;
  /** Muted label, for a reference row such as the baseline. */
  quiet?: boolean;
  /** Shown after the value in muted ink ("+12%", "± 2.1"). */
  tail?: string;
  tip: string[];
}

export interface BarGroup {
  key?: string;
  /** Heading above the group's rows; omit for a flat chart. */
  label?: string;
  rows: BarRow[];
}

/** A row still to come in a live run, drawn as a placeholder under the measured ones. */
export interface PendingBar {
  label: string;
  color: string;
  status: string;
  active: boolean;
}

export interface BarRowsProps {
  groups: BarGroup[];
  pending?: PendingBar[];
  /** Axis maximum; pass a rounded one (niceMax) or a fixed scale such as 100. */
  max: number;
  format: (value: number) => string;
  tickFormat?: (tick: number) => string;
  axisLabel: string;
  /** Dashed reference line, e.g. the baseline's mean. */
  marker?: number | null;
  ink: Ink;
  /** Hover, in body coordinates. */
  onTip: (tip: Tip | null) => void;
}

export function barRowsHeight(
  groups: BarGroup[],
  pending: PendingBar[] = [],
): number {
  return layoutBars(
    groups.map((g) => ({ labelled: !!g.label, rows: g.rows.length })),
    pending.length,
  ).plotH;
}

/** The plot body as an SVG group, for a chart that owns its own frame. */
export function BarRows({
  groups,
  pending = [],
  max,
  format,
  tickFormat = tickText,
  axisLabel,
  marker = null,
  ink: { ink, muted, grid },
  onTip,
}: BarRowsProps): ReactElement {
  const layout = layoutBars(
    groups.map((g) => ({ labelled: !!g.label, rows: g.rows.length })),
    pending.length,
  );
  const { rowH: ROW_H, barH: BAR_H, fontSize: FS, bottom } = layout;
  const plotW = W - LABEL_W - RIGHT_W;
  const x = (v: number) => LABEL_W + (v / max) * plotW;
  const cap = (v: number) => Math.min(max, Math.max(0, v));
  return (
    <g>
      {ticks(max).map((t) => (
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
            {tickFormat(t)}
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
        {axisLabel}
      </text>
      {groups.map((g, gi) => {
        const headTop = layout.headTops[gi];
        return (
          <Fragment key={g.key ?? g.label ?? gi}>
            {headTop !== null && g.label && (
              <text
                x={24}
                y={headTop + 20}
                fontSize={12.5}
                fontWeight={650}
                fill={ink}
              >
                {clip(g.label, 60)}
              </text>
            )}
            {g.rows.map((r, i) => {
              const top = layout.rowTops[gi][i];
              const y = top + (ROW_H - BAR_H) / 2;
              const cy = y + BAR_H / 2;
              const value = cap(r.value);
              const whisker = clampWhisker(r.low, r.high, max);
              const end = cap(Math.max(r.value, r.high ?? r.value));
              return (
                <g
                  key={r.key ?? r.label}
                  onMouseEnter={() => onTip({ x: x(value), y: cy, lines: r.tip })}
                  onMouseLeave={() => onTip(null)}
                >
                  <rect
                    x={0}
                    y={top}
                    width={W}
                    height={ROW_H}
                    fill="transparent"
                  />
                  <text
                    x={LABEL_W - 14}
                    y={cy + 4.5}
                    fontSize={FS}
                    textAnchor="end"
                    fill={r.quiet ? muted : ink}
                  >
                    {clip(r.label, 40)}
                  </text>
                  <path
                    d={barPath(
                      LABEL_W,
                      Math.max(LABEL_W + 2, x(value)),
                      y,
                      BAR_H,
                      4,
                    )}
                    fill={r.color}
                  />
                  {whisker && (
                    <g stroke={ink} strokeOpacity={0.55} strokeWidth={1.5}>
                      <line
                        x1={x(whisker[0])}
                        x2={x(whisker[1])}
                        y1={cy}
                        y2={cy}
                      />
                      <line
                        x1={x(whisker[0])}
                        x2={x(whisker[0])}
                        y1={cy - 4}
                        y2={cy + 4}
                      />
                      <line
                        x1={x(whisker[1])}
                        x2={x(whisker[1])}
                        y1={cy - 4}
                        y2={cy + 4}
                      />
                    </g>
                  )}
                  <text x={x(end) + 12} y={cy + 4.5} fontSize={FS} fill={ink}>
                    <tspan fontWeight={650}>{format(r.value)}</tspan>
                    {r.tail !== undefined && (
                      <tspan dx={8} fill={muted}>
                        {r.tail}
                      </tspan>
                    )}
                  </text>
                </g>
              );
            })}
          </Fragment>
        );
      })}
      {pending.map((p, j) => {
        const y = layout.pendingTops[j] + (ROW_H - BAR_H) / 2;
        const cy = y + BAR_H / 2;
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
              fill={p.active ? p.color : grid}
              fillOpacity={p.active ? 0.16 : 0.35}
            />
            {p.active && (
              <rect
                x={LABEL_W}
                y={y}
                width={4}
                height={BAR_H}
                rx={2}
                fill={p.color}
              />
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
      {marker !== null && (
        <g>
          <line
            x1={x(marker)}
            x2={x(marker)}
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
}

/** A standalone bar chart on the Benchmarks chart card. */
export const BarChart = memo(function BarChart({
  title,
  subtitle,
  footer = [],
  bare = true,
  svgRef,
  ...rows
}: Omit<BarRowsProps, "ink" | "onTip"> & {
  /** Also the SVG's accessible name. */
  title: string;
  subtitle?: string;
  footer?: string[];
  /** On screen: no title, subtitle or fine print. The page around it already says them. */
  bare?: boolean;
  svgRef?: RefObject<SVGSVGElement | null>;
}): ReactElement {
  const ink = useInk();
  const [tip, setTip] = useState<Tip | null>(null);
  const head = chartHead(bare, subtitle);
  const plotH = barRowsHeight(rows.groups, rows.pending);
  const footLines = footerLines(bare, footer);
  return (
    <ChartFrame
      title={title}
      subtitle={subtitle}
      footLines={footLines}
      bare={bare}
      head={head}
      plotH={plotH}
      height={chartHeight(head, plotH, footLines)}
      card={ink.card}
      ink={ink.ink}
      muted={ink.muted}
      svgRef={svgRef}
      tip={tip}
      onMouseLeave={() => setTip(null)}
    >
      <BarRows
        {...rows}
        ink={ink}
        onTip={(t) => setTip(t && { ...t, y: head + t.y })}
      />
    </ChartFrame>
  );
});
