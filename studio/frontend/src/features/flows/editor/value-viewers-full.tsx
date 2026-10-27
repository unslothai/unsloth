// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { toast } from "sonner";
import { fetchMediaBlob, mediaPath } from "../api";
import type { MediaValue } from "../types";
import {
  type ChartValue,
  type HeatmapValue,
  type StructuredValue,
  type TableValue,
  type TensorValue,
  columnLabel,
  formatCell,
  heatColor,
  numericChartX,
  numericStats,
  sortRows,
  tableRows,
  tableToCsv,
  tableToMarkdown,
  tensorAsHeatmap,
} from "./value-kinds";

const SERIES_COLORS = [
  "#22d3ee",
  "#f472b6",
  "#facc15",
  "#a3e635",
  "#a78bfa",
  "#fb923c",
];
const MAX_TABLE_ROWS = 500;

async function copyText(text: string, what: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    toast.success(`Copied ${what}`);
  } catch {
    toast.error("Could not copy to the clipboard");
  }
}

async function fullTable(table: TableValue): Promise<TableValue | null> {
  if (!table.ref || (table.row_count ?? 0) <= table.rows.length) return table;
  try {
    const full = await loadRef<TableValue>(table.ref);
    return { ...table, rows: full.rows ?? table.rows };
  } catch {
    toast.error("Could not load the whole table");
    return null;
  }
}

export function TableViewer({
  value,
}: { value: StructuredValue }): ReactElement | null {
  const table = value as TableValue;
  const [sort, setSort] = useState<{ id: string; desc: boolean } | null>(null);
  const rows = useMemo(() => {
    const all = tableRows(table);
    return sort ? sortRows(all, sort.id, sort.desc) : all;
  }, [table, sort]);
  const total = table.row_count ?? table.rows.length;
  return (
    <div className="flex flex-col gap-1.5" data-testid="flows-table-viewer">
      <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
        <span>
          {total} row{total === 1 ? "" : "s"}
          {total > rows.length ? ` (showing ${rows.length})` : ""}
        </span>
        <span className="flex-1" />
        <button
          type="button"
          className="rounded px-1.5 py-0.5 hover:bg-accent hover:text-foreground"
          onClick={() =>
            void fullTable(table).then((t) =>
              t ? copyText(tableToCsv(t), "CSV") : undefined,
            )
          }
        >
          Copy CSV
        </button>
        <button
          type="button"
          className="rounded px-1.5 py-0.5 hover:bg-accent hover:text-foreground"
          onClick={() =>
            void fullTable(table).then((t) =>
              t ? copyText(tableToMarkdown(t), "Markdown") : undefined,
            )
          }
        >
          Copy Markdown
        </button>
      </div>
      <div className="max-h-80 overflow-auto rounded-md border border-border">
        <table className="w-full border-collapse text-[11.5px]">
          <thead className="sticky top-0 bg-card">
            <tr>
              {table.columns.map((column) => (
                <th
                  key={column.id}
                  className="whitespace-nowrap border-border border-b px-2 py-1 text-left font-medium text-foreground"
                >
                  <button
                    type="button"
                    className="select-none hover:text-foreground"
                    onClick={() =>
                      setSort((s) =>
                        s?.id === column.id
                          ? { id: column.id, desc: !s.desc }
                          : { id: column.id, desc: column.type === "number" },
                      )
                    }
                  >
                    {columnLabel(column)}
                    {sort?.id === column.id ? (sort.desc ? " v" : " ^") : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, MAX_TABLE_ROWS).map((row, i) => (
              <tr
                // biome-ignore lint/suspicious/noArrayIndexKey: rows have no identity of their own
                key={i}
                className="odd:bg-muted/40"
              >
                {table.columns.map((column) => (
                  <td
                    key={column.id}
                    className={`whitespace-nowrap px-2 py-0.5 text-foreground ${
                      column.type === "number" ? "text-right font-mono" : ""
                    }`}
                  >
                    {formatCell(row[column.id])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ChartViewer({
  value,
}: { value: StructuredValue }): ReactElement | null {
  const chart = value as ChartValue;
  const numericX = numericChartX(chart);
  const data = useMemo(
    () =>
      chart.x.map((x, i) => {
        const point: Record<string, unknown> = {
          x: numericX ? x : formatCell(x),
        };
        chart.series.forEach((s, j) => {
          point[`s${j}`] = s.y[i];
          if (s.error) point[`e${j}`] = s.error[i];
        });
        return point;
      }),
    [chart, numericX],
  );
  const yLabel = chart.y_label
    ? `${chart.y_label}${chart.unit ? ` (${chart.unit})` : ""}`
    : chart.unit;
  const axes = [
    <CartesianGrid key="grid" strokeDasharray="3 3" stroke="var(--border)" />,
    <XAxis
      key="x"
      dataKey="x"
      {...(numericX
        ? {
            type: "number" as const,
            domain: ["dataMin", "dataMax"],
            tickFormatter: (v: unknown) => formatCell(v),
          }
        : {})}
      tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
      label={
        chart.x_label
          ? {
              value: chart.x_label,
              position: "insideBottom",
              offset: -2,
              fill: "var(--muted-foreground)",
              fontSize: 11,
            }
          : undefined
      }
    />,
    <YAxis
      key="y"
      tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
      width={52}
      label={
        yLabel
          ? {
              value: yLabel,
              angle: -90,
              position: "insideLeft",
              fill: "var(--muted-foreground)",
              fontSize: 11,
            }
          : undefined
      }
    />,
    <Tooltip
      key="tip"
      contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", color: "var(--popover-foreground)" }}
    />,
    chart.series.length > 1 ? <Legend key="legend" /> : null,
  ];
  return (
    <div className="h-64 w-full" data-testid="flows-chart-viewer">
      <ResponsiveContainer width="100%" height="100%">
        {chart.type === "line" ? (
          <LineChart data={data} margin={{ top: 8, right: 8, bottom: 12 }}>
            {axes}
            {chart.series.map((s, j) => (
              <Line
                key={s.name}
                dataKey={`s${j}`}
                name={s.name}
                stroke={SERIES_COLORS[j % SERIES_COLORS.length]}
                dot={data.length <= 24}
                isAnimationActive={false}
              >
                {s.error ? (
                  <ErrorBar dataKey={`e${j}`} stroke="var(--muted-foreground)" />
                ) : null}
              </Line>
            ))}
          </LineChart>
        ) : (
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 12 }}>
            {axes}
            {chart.series.map((s, j) => (
              <Bar
                key={s.name}
                dataKey={`s${j}`}
                name={s.name}
                fill={SERIES_COLORS[j % SERIES_COLORS.length]}
                isAnimationActive={false}
              >
                {s.error ? (
                  <ErrorBar dataKey={`e${j}`} stroke="var(--muted-foreground)" />
                ) : null}
              </Bar>
            ))}
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

async function loadRef<T>(ref: string, signal?: AbortSignal): Promise<T> {
  const path = mediaPath({ ref } as MediaValue);
  if (!path) throw new Error("Unreadable reference");
  const blob = await fetchMediaBlob(path, signal);
  return JSON.parse(await blob.text()) as T;
}

function useRefValue<T>(ref: string | undefined): T | null {
  const [loaded, setLoaded] = useState<{ ref: string; value: T } | null>(null);
  useEffect(() => {
    if (!ref) return;
    const abort = new AbortController();
    loadRef<T>(ref, abort.signal)
      .then((value) => setLoaded({ ref, value }))
      .catch(() => undefined);
    return () => abort.abort();
  }, [ref]);
  return loaded && loaded.ref === ref ? loaded.value : null;
}

function useMatrix(value: HeatmapValue): (number | null)[][] | null {
  const loaded = useRefValue<{ values?: (number | null)[][] }>(
    value.values ? undefined : value.ref,
  );
  if (value.values) return value.values;
  return Array.isArray(loaded?.values) ? loaded.values : null;
}

function HeatmapCanvas({ value }: { value: HeatmapValue }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const matrix = useMatrix(value);
  const [hover, setHover] = useState<string | null>(null);
  const stats = useMemo(() => numericStats(matrix), [matrix]);
  const lo = value.vmin ?? stats?.min ?? 0;
  const hi = value.vmax ?? stats?.max ?? 1;
  useEffect(() => {
    const el = canvas.current;
    if (!el || !matrix) return;
    const h = matrix.length;
    const w = matrix[0]?.length ?? 0;
    el.width = w;
    el.height = h;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        ctx.fillStyle = heatColor(
          matrix[r][c] ?? null,
          lo,
          hi,
          !!value.diverging,
        );
        ctx.fillRect(c, r, 1, 1);
      }
    }
  }, [matrix, lo, hi, value.diverging]);
  if (!matrix)
    return <p className="text-[11px] text-muted-foreground">Loading heatmap...</p>;
  return (
    <div className="flex flex-col gap-1" data-testid="flows-heatmap-viewer">
      <canvas
        ref={canvas}
        className="max-h-80 w-full rounded border border-border"
        style={{ imageRendering: "pixelated", aspectRatio: "auto" }}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const w = matrix[0]?.length ?? 0;
          const c = Math.floor(((e.clientX - rect.left) / rect.width) * w);
          const r = Math.floor(
            ((e.clientY - rect.top) / rect.height) * matrix.length,
          );
          const v = matrix[r]?.[c];
          setHover(
            v === undefined
              ? null
              : `${value.rows[r] ?? r}, ${value.cols[c] ?? c}: ${v === null ? "missing" : formatCell(v)}`,
          );
        }}
        onMouseLeave={() => setHover(null)}
      />
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span>
          {formatCell(lo)} to {formatCell(hi)}
        </span>
        <span>{hover ?? `${matrix.length} x ${matrix[0]?.length ?? 0}`}</span>
      </div>
    </div>
  );
}

export function HeatmapViewer({
  value,
}: { value: StructuredValue }): ReactElement | null {
  return <HeatmapCanvas value={value as HeatmapValue} />;
}

export function TensorViewer({
  value,
}: { value: StructuredValue }): ReactElement | null {
  const given = value as TensorValue;
  const loaded = useRefValue<{ data?: unknown[] }>(
    given.data ? undefined : given.ref,
  );
  const tensor = useMemo(
    () =>
      given.data || !Array.isArray(loaded?.data)
        ? given
        : { ...given, data: loaded.data },
    [given, loaded],
  );
  const heatmap = useMemo(() => tensorAsHeatmap(tensor), [tensor]);
  const stats = useMemo(() => numericStats(tensor.data), [tensor]);
  return (
    <div className="flex flex-col gap-1.5" data-testid="flows-tensor-viewer">
      <p className="font-mono text-[11px] text-muted-foreground">
        shape [{tensor.shape.join(", ")}]
        {tensor.dtype ? ` ${tensor.dtype}` : ""}
        {tensor.axes?.length ? ` (${tensor.axes.join(", ")})` : ""}
        {stats
          ? `  min ${formatCell(stats.min)}  max ${formatCell(stats.max)}  mean ${formatCell(stats.mean)}`
          : ""}
      </p>
      {heatmap ? <HeatmapCanvas value={heatmap} /> : null}
    </div>
  );
}
