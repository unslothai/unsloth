// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export interface TableColumn {
  id: string;
  label?: string;
  type?: "number" | "text" | "bool" | string;
  unit?: string;
  role?: "dim" | "metric" | "error" | "delta" | string;
}

export interface TableValue {
  kind: "table";
  columns: TableColumn[];
  rows: Record<string, unknown>[] | unknown[][];
  meta?: Record<string, unknown>;
  row_count?: number;
  ref?: string;
}

export interface ChartSeries {
  name: string;
  y: (number | null)[];
  error?: (number | null)[];
}

export interface ChartValue {
  kind: "chart";
  type: "bar" | "line";
  x: unknown[];
  series: ChartSeries[];
  x_label?: string;
  y_label?: string | null;
  unit?: string;
}

export interface HeatmapValue {
  kind: "heatmap";
  rows: string[];
  cols: string[];
  values?: (number | null)[][];
  ref?: string;
  vmin?: number;
  vmax?: number;
  diverging?: boolean;
}

export interface TensorValue {
  kind: "tensor";
  shape: number[];
  dtype?: string;
  axes?: string[];
  data?: unknown[];
  ref?: string;
}

export type StructuredValue =
  | TableValue
  | ChartValue
  | HeatmapValue
  | TensorValue;
export type StructuredKind = StructuredValue["kind"];

export const STRUCTURED_KINDS: readonly StructuredKind[] = [
  "table",
  "chart",
  "heatmap",
  "tensor",
];

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const isColumn = (c: unknown) => isObject(c) && typeof c.id === "string";
const isRow = (r: unknown) => isObject(r) || Array.isArray(r);
const isSeries = (s: unknown) =>
  isObject(s) &&
  Array.isArray(s.y) &&
  (s.error === undefined || Array.isArray(s.error));
const isMatrix = (m: unknown) => Array.isArray(m) && m.every(Array.isArray);

export function structuredKind(value: unknown): StructuredKind | null {
  if (!isObject(value)) return null;
  switch (value.kind) {
    case "table":
      return Array.isArray(value.columns) &&
        value.columns.every(isColumn) &&
        Array.isArray(value.rows) &&
        value.rows.every(isRow)
        ? "table"
        : null;
    case "chart":
      return Array.isArray(value.x) &&
        Array.isArray(value.series) &&
        value.series.every(isSeries)
        ? "chart"
        : null;
    case "heatmap":
      return Array.isArray(value.rows) &&
        Array.isArray(value.cols) &&
        (isMatrix(value.values) || typeof value.ref === "string")
        ? "heatmap"
        : null;
    case "tensor":
      return Array.isArray(value.shape) &&
        (value.data === undefined || Array.isArray(value.data))
        ? "tensor"
        : null;
    default:
      return null;
  }
}

export function tableRows(table: TableValue): Record<string, unknown>[] {
  return table.rows.map((row) =>
    Array.isArray(row)
      ? Object.fromEntries(table.columns.map((c, i) => [c.id, row[i]]))
      : row,
  );
}

export function columnLabel(column: TableColumn): string {
  const label = column.label || column.id;
  return column.unit ? `${label} (${column.unit})` : label;
}

export function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "";
    if (Number.isInteger(value)) return String(value);
    return Math.abs(value) >= 100 ? value.toFixed(1) : value.toPrecision(4);
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function sortRows(
  rows: Record<string, unknown>[],
  column: string,
  descending: boolean,
): Record<string, unknown>[] {
  const sign = descending ? -1 : 1;
  const asNumber = (v: unknown): number =>
    typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  const numeric = rows.every((row) => {
    const v = row[column];
    return v === null || v === undefined || Number.isFinite(asNumber(v));
  });
  return rows
    .map((row, index) => ({ row, index }))
    .sort((a, b) => {
      const x = a.row[column];
      const y = b.row[column];
      const xEmpty = x === null || x === undefined;
      const yEmpty = y === null || y === undefined;
      if (xEmpty || yEmpty)
        return xEmpty === yEmpty ? a.index - b.index : xEmpty ? 1 : -1;
      const order = numeric
        ? asNumber(x) - asNumber(y)
        : String(x).localeCompare(String(y));
      return order === 0 ? a.index - b.index : sign * order;
    })
    .map((entry) => entry.row);
}

function csvCell(value: unknown): string {
  const text =
    typeof value === "number"
      ? Number.isFinite(value)
        ? String(value)
        : ""
      : formatCell(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function tableToCsv(table: TableValue): string {
  const head = table.columns.map((c) => csvCell(columnLabel(c))).join(",");
  const body = tableRows(table).map((row) =>
    table.columns.map((c) => csvCell(row[c.id])).join(","),
  );
  return [head, ...body].join("\n");
}

function mdCell(value: string): string {
  return value.replace(/\|/g, "\\|").replace(/\r?\n/g, " ");
}

export function tableToMarkdown(table: TableValue): string {
  const head = `| ${table.columns.map((c) => mdCell(columnLabel(c))).join(" | ")} |`;
  const rule = `| ${table.columns
    .map((c) => (c.type === "number" ? "---:" : "---"))
    .join(" | ")} |`;
  const body = tableRows(table).map(
    (row) =>
      `| ${table.columns.map((c) => mdCell(formatCell(row[c.id]))).join(" | ")} |`,
  );
  return [head, rule, ...body].join("\n");
}

export function structuredSummary(value: StructuredValue): string {
  switch (value.kind) {
    case "table": {
      const rows = value.row_count ?? value.rows.length;
      return `Table: ${rows} row${rows === 1 ? "" : "s"} x ${value.columns.length} column${value.columns.length === 1 ? "" : "s"}`;
    }
    case "chart":
      return `${value.type === "line" ? "Line" : "Bar"} chart: ${value.series.length} series, ${value.x.length} points`;
    case "heatmap":
      return `Heatmap: ${value.rows.length} x ${value.cols.length}`;
    case "tensor":
      return `Tensor ${value.shape.join(" x ")}${value.dtype ? ` ${value.dtype}` : ""}`;
  }
}

export function numericStats(
  data: unknown,
): { min: number; max: number; mean: number; count: number } | null {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let count = 0;
  const stack: unknown[] = [data];
  while (stack.length) {
    const item = stack.pop();
    if (Array.isArray(item)) {
      for (const child of item) stack.push(child);
    } else if (typeof item === "number" && Number.isFinite(item)) {
      if (item < min) min = item;
      if (item > max) max = item;
      sum += item;
      count += 1;
    }
  }
  return count ? { min, max, mean: sum / count, count } : null;
}

export function tensorAsHeatmap(tensor: TensorValue): HeatmapValue | null {
  if (tensor.shape.length !== 2 || !Array.isArray(tensor.data)) return null;
  const [h, w] = tensor.shape;
  const data = tensor.data;
  const rows: (number | null)[][] = [];
  const nested = Array.isArray(data[0]);
  for (let r = 0; r < h; r++) {
    const row: (number | null)[] = [];
    for (let c = 0; c < w; c++) {
      const v = nested
        ? (data[r] as unknown[] | undefined)?.[c]
        : data[r * w + c];
      row.push(typeof v === "number" && Number.isFinite(v) ? v : null);
    }
    rows.push(row);
  }
  const axes = tensor.axes ?? [];
  return {
    kind: "heatmap",
    rows: Array.from({ length: h }, (_, i) => `${axes[0] ?? "row"} ${i}`),
    cols: Array.from({ length: w }, (_, i) => `${axes[1] ?? "col"} ${i}`),
    values: rows,
  };
}

export function heatColor(
  value: number | null,
  lo: number,
  hi: number,
  diverging: boolean,
): string {
  if (value === null || !Number.isFinite(value)) return "rgba(0,0,0,0)";
  const span = hi - lo || 1;
  const t = Math.min(1, Math.max(0, (value - lo) / span));
  if (diverging) {
    const d = t * 2 - 1;
    const a = Math.round(255 * (1 - Math.abs(d)));
    return d < 0 ? `rgb(${a},${a},255)` : `rgb(255,${a},${a})`;
  }
  const r = Math.round(20 + 235 * t);
  const g = Math.round(20 + 180 * t ** 1.5);
  const b = Math.round(60 + 60 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

export function numericChartX(chart: Pick<ChartValue, "type" | "x">): boolean {
  return (
    chart.type === "line" &&
    chart.x.length > 0 &&
    chart.x.every((x) => typeof x === "number" && Number.isFinite(x))
  );
}

/** The output ids a card draws: its ports, or for a sink card with none (Preview, Result) what the run returned. */
export function shownOutputIds(
  ports: readonly { id: string }[],
  outputs: Record<string, unknown>,
): string[] {
  return ports.length ? ports.map((port) => port.id) : Object.keys(outputs);
}

/** The text a copy button takes from a step's outputs: the first shown text, list or object;
 *  null when the step only made media or nothing. */
export function outputCopyText(
  ports: readonly { id: string }[],
  outputs: Record<string, unknown> | null | undefined,
): string | null {
  if (!outputs) return null;
  for (const id of shownOutputIds(ports, outputs)) {
    const value = outputs[id];
    if (value === undefined || value === null) continue;
    if (typeof value === "string") return value || null;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    if (isMediaLike(value) || (Array.isArray(value) && value.some(isMediaLike))) continue;
    if (Array.isArray(value))
      return value
        .map((item) => (typeof item === "string" ? item : JSON.stringify(item, null, 2)))
        .join("\n");
    return JSON.stringify(value, null, 2);
  }
  return null;
}

function isMediaLike(value: unknown): boolean {
  return (
    !!value &&
    typeof value === "object" &&
    typeof (value as { ref?: unknown }).ref === "string" &&
    typeof (value as { kind?: unknown }).kind === "string"
  );
}
