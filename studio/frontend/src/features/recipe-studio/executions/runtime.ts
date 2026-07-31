// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { JobEvent, JobStatusResponse } from "../api";
import type {
  RecipeExecutionBatch,
  RecipeExecutionKind,
  RecipeExecutionRecord,
  RecipeSourceProgress,
} from "../execution-types";
import {
  DATASET_PAGE_SIZE,
  mapJobStatus,
  normalizeObject,
} from "./execution-helpers";

const MAX_LOG_LINES = 1500;

function formatEventTime(ts: unknown): string {
  if (typeof ts !== "number" || !Number.isFinite(ts)) {
    return new Date().toLocaleTimeString();
  }
  const ms = ts > 10_000_000_000 ? ts : ts * 1000;
  return new Date(ms).toLocaleTimeString();
}

export function appendExecutionLogLine(lines: string[], nextLine: string): string[] {
  const next = [...lines, nextLine];
  if (next.length <= MAX_LOG_LINES) {
    return next;
  }
  return next.slice(next.length - MAX_LOG_LINES);
}

export function toExecutionLogLine(event: JobEvent): string | null {
  const eventType =
    typeof event.payload.type === "string" ? event.payload.type : event.event;
  const ts = formatEventTime(event.payload.ts);

  if (eventType === "log") {
    const message =
      typeof event.payload.message === "string" ? event.payload.message.trim() : "";
    if (!message) {
      return null;
    }
    const level =
      typeof event.payload.level === "string" && event.payload.level.length > 0
        ? event.payload.level.toUpperCase()
        : "INFO";
    return `[${ts}] [${level}] ${message}`;
  }

  if (eventType === "job.started") {
    return `[${ts}] [INFO] Job started`;
  }
  if (eventType === "job.completed") {
    return `[${ts}] [INFO] Job completed`;
  }
  if (eventType === "job.cancelling") {
    return `[${ts}] [WARN] Cancellation requested`;
  }
  if (eventType === "job.cancelled") {
    return `[${ts}] [WARN] Job cancelled`;
  }
  if (eventType === "job.error") {
    const error =
      typeof event.payload.error === "string" && event.payload.error.length > 0
        ? event.payload.error
        : "Job failed";
    return `[${ts}] [ERROR] ${error}`;
  }

  return null;
}

function normalizeSourceProgress(input: unknown): RecipeSourceProgress | null {
  const raw = normalizeObject(input);
  if (!raw) {
    return null;
  }
  return raw as RecipeSourceProgress;
}

export function applyExecutionStatusSnapshot(
  execution: RecipeExecutionRecord,
  status: JobStatusResponse,
): RecipeExecutionRecord {
  const mappedStatus = mapJobStatus(status.status);
  const batchRaw = normalizeObject(status.batch);
  const batch: RecipeExecutionBatch | null = batchRaw
    ? {
        idx: typeof batchRaw.idx === "number" ? batchRaw.idx : null,
        total: typeof batchRaw.total === "number" ? batchRaw.total : null,
      }
    : null;
  return {
    ...execution,
    status: mappedStatus,
    rows: status.rows ?? execution.rows,
    stage: status.stage ?? execution.stage,
    current_column: status.current_column ?? null,
    completed_columns: Array.isArray(status.completed_columns)
      ? status.completed_columns.filter(
          (value): value is string => typeof value === "string" && value.trim().length > 0,
        )
      : execution.completed_columns,
    progress: (normalizeObject(status.progress) as RecipeExecutionRecord["progress"]) ?? null,
    column_progress:
      (normalizeObject(status.column_progress) as RecipeExecutionRecord["column_progress"]) ??
      null,
    batch,
    source_progress: normalizeSourceProgress(status.source_progress),
    model_usage: normalizeObject(status.model_usage),
    artifact_path: status.artifact_path ?? execution.artifact_path,
    error: status.error ?? null,
    finishedAt:
      mappedStatus === "completed" ||
      mappedStatus === "error" ||
      mappedStatus === "cancelled"
        ? Date.now()
        : null,
  };
}

export function createBaseExecutionRecord(input: {
  recipeId: string;
  kind: RecipeExecutionKind;
  rows: number;
  currentSignature: string;
  runName?: string | null;
}): RecipeExecutionRecord {
  const createdAt = Date.now();
  return {
    id: crypto.randomUUID(),
    recipeId: input.recipeId,
    jobId: null,
    kind: input.kind,
    run_name: input.runName ?? null,
    status: "pending",
    rows: input.rows,
    createdAt,
    finishedAt: null,
    recipeSignature: input.currentSignature,
    stage: "pending",
    current_column: null,
    completed_columns: [],
    progress: null,
    column_progress: null,
    batch: null,
    source_progress: null,
    model_usage: null,
    lastEventId: null,
    artifact_path: null,
    log_lines: [],
    dataset: [],
    datasetTotal: 0,
    datasetPage: 1,
    datasetPageSize: DATASET_PAGE_SIZE,
    analysis: null,
    processor_artifacts: null,
    error: null,
  };
}
