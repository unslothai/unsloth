// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  RecipeExecutionAnalysis,
  RecipeExecutionRecord,
  RecipeExecutionStatus,
} from "../execution-types";
import type { RecipePayload } from "../utils/payload/types";

export const DATASET_PAGE_SIZE = 20;

export function buildSignature(name: string, payload: RecipePayload): string {
  return JSON.stringify({ name, payload });
}

export function formatSavedLabel(savedAt: number | null): string {
  if (!savedAt) {
    return "Not saved yet";
  }
  const time = new Date(savedAt).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
  return `Saved ${time}`;
}

export function toErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error) {
    return error.message;
  }
  return fallback;
}

export function normalizeDatasetRows(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (row): row is Record<string, unknown> =>
      typeof row === "object" && row !== null && !Array.isArray(row),
  );
}

export function normalizeObject(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

export function normalizeAnalysis(value: unknown): RecipeExecutionAnalysis | null {
  const normalized = normalizeObject(value);
  if (!normalized) {
    return null;
  }
  return normalized as RecipeExecutionAnalysis;
}

export function mapJobStatus(status: string): RecipeExecutionStatus {
  if (status === "active") {
    return "active";
  }
  if (status === "pending") {
    return "pending";
  }
  if (status === "cancelling") {
    return "cancelling";
  }
  if (status === "cancelled") {
    return "cancelled";
  }
  if (status === "completed") {
    return "completed";
  }
  if (status === "error") {
    return "error";
  }
  return "running";
}

export function isExecutionInProgress(status: RecipeExecutionStatus): boolean {
  return (
    status === "running" ||
    status === "active" ||
    status === "pending" ||
    status === "cancelling"
  );
}

export function executionLabel(kind: "preview" | "full"): string {
  return kind === "preview" ? "Preview" : "Full run";
}

export function normalizeRunName(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function executionSortWeight(status: RecipeExecutionStatus): number {
  if (isExecutionInProgress(status)) {
    return 0;
  }
  if (status === "error" || status === "cancelled") {
    return 2;
  }
  return 1;
}

export function sortExecutions(records: RecipeExecutionRecord[]): RecipeExecutionRecord[] {
  const next = [...records];
  next.sort((a, b) => {
    const statusDelta = executionSortWeight(a.status) - executionSortWeight(b.status);
    if (statusDelta !== 0) {
      return statusDelta;
    }
    return b.createdAt - a.createdAt;
  });
  return next;
}

export function withExecutionDefaults(
  record: RecipeExecutionRecord,
): RecipeExecutionRecord {
  const dataset = Array.isArray(record.dataset) ? record.dataset : [];
  const logLines = Array.isArray(record.log_lines)
    ? record.log_lines.filter((line): line is string => typeof line === "string")
    : [];
  const datasetPageSize =
    typeof record.datasetPageSize === "number" && record.datasetPageSize > 0
      ? record.datasetPageSize
      : DATASET_PAGE_SIZE;
  const datasetPage =
    typeof record.datasetPage === "number" && record.datasetPage > 0
      ? record.datasetPage
      : 1;
  const datasetTotal =
    typeof record.datasetTotal === "number" && record.datasetTotal >= 0
      ? record.datasetTotal
      : dataset.length;

  return {
    ...record,
    run_name: normalizeRunName(record.run_name),
    dataset,
    log_lines: logLines,
    datasetTotal,
    datasetPage,
    datasetPageSize,
    completed_columns: Array.isArray(record.completed_columns)
      ? record.completed_columns.filter(
          (value): value is string => typeof value === "string" && value.trim().length > 0,
        )
      : [],
    column_progress: record.column_progress ?? null,
    batch: record.batch ?? null,
  };
}

export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

export async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fallthrough to legacy path
  }

  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "0";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(textarea);
    return ok;
  } catch {
    return false;
  }
}
