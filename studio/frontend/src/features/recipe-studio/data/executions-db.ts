// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type RecipeExecutionPage,
  UserAssetApiError,
  listServerRecipeExecutions,
  upsertServerRecipeExecution,
} from "@/features/user-assets";
import type {
  PersistedRecipeExecution,
  RecipeExecutionMetadata,
  RecipeExecutionRecord,
  RecipeExecutionStatus,
} from "../execution-types";

const revisions = new Map<string, number>();
const writeQueues = new Map<string, Promise<void>>();
const MAX_EXECUTION_JSON_BYTES = 256 * 1024;
const TERMINAL = new Set<RecipeExecutionStatus>([
  "cancelled",
  "completed",
  "error",
]);

function truncateUtf8(value: unknown, maxBytes: number): string | null {
  if (typeof value !== "string") return null;
  const encoder = new TextEncoder();
  if (encoder.encode(value).byteLength <= maxBytes) return value;
  let result = "";
  let byteLength = 0;
  for (const codePoint of value) {
    const codePointBytes = encoder.encode(codePoint).byteLength;
    if (byteLength + codePointBytes > maxBytes) break;
    result += codePoint;
    byteLength += codePointBytes;
  }
  return result;
}

function jsonByteLength(value: unknown): number {
  return new TextEncoder().encode(JSON.stringify(value)).byteLength;
}

function boundedObject<T extends object>(
  value: T | null,
  maxBytes: number,
): T | null {
  if (!value) return null;
  try {
    const encoded = JSON.stringify(value);
    if (new TextEncoder().encode(encoded).byteLength > maxBytes) return null;
    return JSON.parse(encoded) as T;
  } catch {
    return null;
  }
}

function fitsExecutionBudget(metadata: RecipeExecutionMetadata): boolean {
  return jsonByteLength(metadata) <= MAX_EXECUTION_JSON_BYTES;
}

function fitCompletedColumns(
  metadata: RecipeExecutionMetadata,
): RecipeExecutionMetadata {
  if (fitsExecutionBudget(metadata)) return metadata;
  let low = 0;
  let high = metadata.completed_columns.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    const candidate = {
      ...metadata,
      completed_columns: metadata.completed_columns.slice(0, middle),
    };
    if (fitsExecutionBudget(candidate)) low = middle;
    else high = middle - 1;
  }
  return {
    ...metadata,
    completed_columns: metadata.completed_columns.slice(0, low),
  };
}

/** The only projection allowed to cross the persistence boundary. */
export function serializeExecutionMetadata(
  record: RecipeExecutionRecord,
): RecipeExecutionMetadata {
  let metadata: RecipeExecutionMetadata = {
    id: record.id,
    recipeId: record.recipeId,
    jobId: truncateUtf8(record.jobId, 128),
    kind: record.kind,
    run_name: truncateUtf8(record.run_name, 200),
    status: record.status,
    rows: Math.max(0, Math.floor(record.rows || 0)),
    recipeSignature: truncateUtf8(record.recipeSignature, 4096) ?? "",
    stage: truncateUtf8(record.stage, 200),
    current_column: truncateUtf8(record.current_column, 200),
    completed_columns: (Array.isArray(record.completed_columns)
      ? record.completed_columns
      : []
    )
      .slice(0, 1000)
      .map((value) => truncateUtf8(value, 200) ?? ""),
    progress: null,
    column_progress: null,
    batch: null,
    source_progress: null,
    model_usage: null,
    lastEventId: record.lastEventId,
    datasetTotal: Math.max(0, Math.floor(record.datasetTotal || 0)),
    analysis: null,
    error: truncateUtf8(record.error, 4096),
    createdAt:
      typeof record.createdAt === "number" && Number.isFinite(record.createdAt)
        ? Math.max(0, Math.floor(record.createdAt))
        : 0,
    finishedAt:
      typeof record.finishedAt === "number" &&
      Number.isFinite(record.finishedAt)
        ? Math.max(0, Math.floor(record.finishedAt))
        : null,
  };

  // Scalars and completed-column names are the durable resume envelope. If a
  // future field expansion ever pushes that envelope over budget, keep the
  // largest deterministic prefix rather than emitting a request the backend
  // must reject.
  metadata = fitCompletedColumns(metadata);

  // Add structured snapshots in resume/display priority order. Each candidate
  // is canonicalized independently, and is omitted wholesale if it would push
  // the complete metadata projection above the backend's 256 KiB cap.
  const optionalObjects = [
    ["progress", boundedObject(record.progress, MAX_EXECUTION_JSON_BYTES)],
    [
      "column_progress",
      boundedObject(record.column_progress, MAX_EXECUTION_JSON_BYTES),
    ],
    ["batch", boundedObject(record.batch, MAX_EXECUTION_JSON_BYTES)],
    [
      "source_progress",
      boundedObject(record.source_progress, MAX_EXECUTION_JSON_BYTES),
    ],
    ["model_usage", boundedObject(record.model_usage, 32 * 1024)],
    ["analysis", boundedObject(record.analysis, 128 * 1024)],
  ] as const;
  for (const [key, value] of optionalObjects) {
    if (value === null) continue;
    const candidate = { ...metadata, [key]: value };
    if (fitsExecutionBudget(candidate)) metadata = candidate;
  }

  if (!fitsExecutionBudget(metadata)) {
    throw new Error("Execution metadata exceeds the persistence limit.");
  }
  return metadata;
}

function incomingWins(
  incoming: RecipeExecutionMetadata,
  current: PersistedRecipeExecution,
): boolean {
  const incomingEvent = incoming.lastEventId ?? -1;
  const currentEvent = current.lastEventId ?? -1;
  return (
    incomingEvent > currentEvent ||
    (TERMINAL.has(incoming.status) && !TERMINAL.has(current.status))
  );
}

async function persistOnce(record: RecipeExecutionRecord): Promise<void> {
  const metadata = serializeExecutionMetadata(record);
  const expectedRevision = revisions.get(record.id) ?? record.revision;
  try {
    const saved = await upsertServerRecipeExecution<PersistedRecipeExecution>({
      recipeId: record.recipeId,
      executionId: record.id,
      metadata,
      revision: expectedRevision,
    });
    revisions.set(record.id, saved.revision);
  } catch (error) {
    if (!(error instanceof UserAssetApiError) || error.status !== 409)
      throw error;
    const current = error.detail.current as
      | PersistedRecipeExecution
      | undefined;
    if (!current) throw error;
    revisions.set(record.id, current.revision);
    if (!incomingWins(metadata, current)) return;
    const saved = await upsertServerRecipeExecution<PersistedRecipeExecution>({
      recipeId: record.recipeId,
      executionId: record.id,
      metadata,
      revision: current.revision,
    });
    revisions.set(record.id, saved.revision);
  }
}

export async function listRecipeExecutions(
  recipeId: string,
): Promise<PersistedRecipeExecution[]> {
  const executions: PersistedRecipeExecution[] = [];
  let cursor: string | null = null;
  do {
    const page: RecipeExecutionPage<PersistedRecipeExecution> =
      await listServerRecipeExecutions<PersistedRecipeExecution>(recipeId, {
        cursor,
        limit: 100,
      });
    executions.push(...page.executions);
    cursor = page.nextCursor;
  } while (cursor);
  for (const execution of executions)
    revisions.set(execution.id, execution.revision);
  return executions;
}

export function saveRecipeExecution(
  execution: RecipeExecutionRecord,
): Promise<void> {
  const previous = writeQueues.get(execution.id) ?? Promise.resolve();
  const next = previous
    .catch(() => undefined)
    .then(() => persistOnce(execution));
  const queued = next.then(
    () => {
      if (writeQueues.get(execution.id) === queued)
        writeQueues.delete(execution.id);
    },
    () => {
      if (writeQueues.get(execution.id) === queued)
        writeQueues.delete(execution.id);
    },
  );
  writeQueues.set(execution.id, queued);
  return next;
}
