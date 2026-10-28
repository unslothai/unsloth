// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DatasetValue } from "../types";

/** The shape `core/flows/dataset_rows.py` returns. */
export interface DatasetRowsPage {
  rows: Record<string, unknown>[];
  total: number | null;
  offset: number;
  view_rows: number;
}

export interface DatasetColumnStat {
  column_name: string;
  column_type: string;
  simple_dtype: string;
  num_unique: number | null;
  num_null: number | null;
  input_tokens_mean: number | null;
  output_tokens_mean: number | null;
}

const SOURCE_LABEL: Record<string, string> = {
  hf: "Hugging Face",
  recipe: "Data Recipe",
  local: "Uploaded",
};

export function isDatasetValue(value: unknown): value is DatasetValue {
  return (
    !!value &&
    typeof value === "object" &&
    (value as { kind?: unknown }).kind === "dataset" &&
    typeof (value as { ref?: unknown }).ref === "string"
  );
}

export function datasetTitle(value: DatasetValue): string {
  if (value.source === "hf") return value.ref;
  const name = value.ref.split("/").filter((p) => p && p !== "parquet-files").pop();
  return value.source === "recipe" || !name || name.startsWith("ref:")
    ? (SOURCE_LABEL[value.source] ?? "Dataset")
    : name;
}

export function datasetSummary(value: DatasetValue, total?: number | null): string {
  const rows = total ?? value.rows;
  const parts = [SOURCE_LABEL[value.source] ?? value.source];
  if (typeof rows === "number") parts.push(`${rows.toLocaleString()} rows`);
  if (value.columns?.length) parts.push(`${value.columns.length} columns`);
  return parts.join(" · ");
}

/** Row count to page by: the reader's total, the value's own count, or as far as rows have come back. */
export function knownTotal(
  value: DatasetValue,
  page: Pick<DatasetRowsPage, "total" | "offset" | "rows"> | null,
  limit: number,
): number | null {
  if (typeof page?.total === "number") return page.total;
  if (typeof value.rows === "number") return value.rows;
  if (page && page.rows.length < limit) return page.offset + page.rows.length;
  return null;
}

export function pageCount(
  total: number | null,
  viewRows: number,
  limit: number,
  offset: number,
): number {
  if (total === null) return Math.floor(offset / limit) + 2;
  return Math.max(1, Math.ceil(Math.min(total, viewRows) / limit));
}

function dtypeOf(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "number")
    return Number.isInteger(value) ? "int" : "float";
  if (typeof value === "boolean") return "bool";
  if (typeof value === "string") return "string";
  if (Array.isArray(value)) return "list";
  return "dict";
}

export function columnNames(rows: Record<string, unknown>[]): string[] {
  const names = new Set<string>();
  for (const row of rows) for (const key of Object.keys(row)) names.add(key);
  return [...names];
}

/** Per-column type, distinct values and nulls over the rows shown, in Recipe Studio's Columns-tab shape. */
export function columnStats(
  rows: Record<string, unknown>[],
): DatasetColumnStat[] {
  return columnNames(rows).map((name) => {
    const types = new Set<string>();
    const seen = new Set<string>();
    let nulls = 0;
    for (const row of rows) {
      const value = row[name];
      const type = dtypeOf(value);
      if (type === "null") nulls += 1;
      else types.add(type);
      seen.add(typeof value === "string" ? value : JSON.stringify(value));
    }
    return {
      column_name: name,
      column_type: "data",
      simple_dtype: types.size ? [...types].join(" | ") : "null",
      num_unique: seen.size,
      num_null: nulls,
      input_tokens_mean: null,
      output_tokens_mean: null,
    };
  });
}

export function columnDtypes(rows: Record<string, unknown>[]): Record<string, string> {
  return Object.fromEntries(
    columnStats(rows).map((c) => [c.column_name, c.simple_dtype]),
  );
}
