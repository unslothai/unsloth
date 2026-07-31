// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  RecipeExecutionAnalysis,
  RecipeExecutionStatus,
} from "../../execution-types";
import { isExecutionInProgress } from "../../executions/execution-helpers";
import { resolveImagePreview } from "../../utils/image-preview";

export type AnalysisColumnStat = {
  column_name: string;
  column_type: string;
  simple_dtype: string;
  num_unique: number | null;
  num_null: number | null;
  input_tokens_mean: number | null;
  output_tokens_mean: number | null;
};

export type ModelUsageRow = {
  model: string;
  input: number | null;
  output: number | null;
};

export const PREVIEW_DATASET_PAGE_SIZE = 20;
export const TERMINAL_STICKY_BOTTOM_THRESHOLD_PX = 24;

export function formatTimestamp(value: number): string {
  return new Date(value).toLocaleString();
}

export function formatCellValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "--";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function isExpandableCellValue(value: string): boolean {
  return value.length > 180;
}

export function truncateCellValue(value: string): string {
  if (value.length <= 180) {
    return value;
  }
  return `${value.slice(0, 180).trimEnd()}...`;
}

export function hasExpandableTextCell(
  row: Record<string, unknown>,
  visibleColumnNames: string[],
): boolean {
  return visibleColumnNames.some((columnName) => {
    if (resolveImagePreview(row[columnName])) {
      return false;
    }
    return isExpandableCellValue(formatCellValue(row[columnName]));
  });
}

function parseNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseString(value: unknown): string {
  return typeof value === "string" && value.length > 0 ? value : "--";
}

export function parseAnalysisColumns(
  analysis: RecipeExecutionAnalysis | null,
): AnalysisColumnStat[] {
  const items = Array.isArray(analysis?.column_statistics)
    ? analysis.column_statistics
    : [];
  return items
    .map((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        return null;
      }
      const row = item as Record<string, unknown>;
      return {
        column_name: parseString(row.column_name),
        column_type: parseString(row.column_type),
        simple_dtype: parseString(row.simple_dtype),
        num_unique: parseNumber(row.num_unique),
        num_null: parseNumber(row.num_null),
        input_tokens_mean: parseNumber(row.input_tokens_mean),
        output_tokens_mean: parseNumber(row.output_tokens_mean),
      };
    })
    .filter((item): item is AnalysisColumnStat => item !== null);
}

export function statusTone(status: RecipeExecutionStatus): string {
  if (status === "completed") {
    return "border-emerald-500/30 text-emerald-700 dark:text-emerald-300";
  }
  if (status === "error" || status === "cancelled") {
    return "border-red-500/30 text-red-700 dark:text-red-300";
  }
  if (isExecutionInProgress(status)) {
    return "border-amber-500/30 text-amber-700 dark:text-amber-300";
  }
  return "border-border/60 text-muted-foreground";
}

export function statusRightBorder(status: RecipeExecutionStatus): string {
  if (status === "completed") {
    return "border-r-emerald-500/40";
  }
  if (status === "error" || status === "cancelled") {
    return "border-r-red-500/40";
  }
  if (isExecutionInProgress(status)) {
    return "border-r-amber-500/40";
  }
  return "border-r-border/50";
}

export function formatStatus(status: RecipeExecutionStatus): string {
  if (status === "cancelled") {
    return "cancelled";
  }
  return status;
}

export function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(1)}%`;
}

export function formatDuration(startedAt: number, finishedAt: number | null): string {
  if (!finishedAt || finishedAt <= startedAt) {
    return "--";
  }
  const seconds = Math.round((finishedAt - startedAt) / 1000);
  return `${seconds}s`;
}

export function formatMetricValue(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return value.toLocaleString();
}

export function parseModelUsageRows(
  value: Record<string, unknown> | null,
): ModelUsageRow[] {
  if (!value) {
    return [];
  }
  return Object.entries(value)
    .map(([name, data]) => {
      if (!data || typeof data !== "object" || Array.isArray(data)) {
        return null;
      }
      const modelObj = data as Record<string, unknown>;
      const tokens =
        modelObj.tokens &&
        typeof modelObj.tokens === "object" &&
        !Array.isArray(modelObj.tokens)
          ? (modelObj.tokens as Record<string, unknown>)
          : null;
      const modelName =
        typeof modelObj.model === "string" && modelObj.model.length > 0
          ? modelObj.model
          : name;
      return {
        model: modelName,
        input: parseNumber(tokens?.input),
        output: parseNumber(tokens?.output),
      };
    })
    .filter((item): item is ModelUsageRow => item !== null);
}
