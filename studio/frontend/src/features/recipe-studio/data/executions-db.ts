// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSubjectKey } from "@/features/auth";
import {
  type RecipeExecutionPage,
  UserAssetApiError,
  listServerRecipeExecutions,
  upsertServerRecipeExecution,
} from "@/features/user-assets";
// Shared persistence policy is infrastructure, not a feature UI dependency.
// eslint-disable-next-line no-restricted-imports
import { MAX_EXECUTION_JSON_BYTES } from "@/features/user-assets/persistence-policy";
import type {
  PersistedRecipeExecution,
  RecipeExecutionMetadata,
  RecipeExecutionRecord,
  RecipeExecutionStatus,
} from "../execution-types";

const revisions = new Map<string, number>();
const persistedExecutions = new Map<string, PersistedRecipeExecution>();
type WriteState = {
  subject: string;
  latest: RecipeExecutionRecord | null;
  running: boolean;
  timer: ReturnType<typeof setTimeout> | null;
  waiters: Array<{ resolve: () => void; reject: (error: unknown) => void }>;
};
const writeStates = new Map<string, WriteState>();
let activeSubjectKey: string | null = null;
const TERMINAL = new Set<RecipeExecutionStatus>([
  "cancelled",
  "completed",
  "error",
]);

function syncSubject(): string {
  const subject = getAuthSubjectKey();
  if (activeSubjectKey !== subject) {
    for (const state of writeStates.values()) {
      if (state.timer) clearTimeout(state.timer);
      state.latest = null;
      for (const waiter of state.waiters.splice(0)) {
        waiter.reject(new Error("Execution persistence account changed."));
      }
    }
    revisions.clear();
    persistedExecutions.clear();
    writeStates.clear();
    activeSubjectKey = subject;
  }
  return subject;
}

function assertSubjectUnchanged(subject: string): void {
  if (getAuthSubjectKey() !== subject) {
    throw new Error("Execution persistence account changed.");
  }
}

function executionKey(subject: string, id: string): string {
  return `${subject}\u0000${id}`;
}

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
  const incomingTerminal = TERMINAL.has(incoming.status);
  const currentTerminal = TERMINAL.has(current.status);
  if (currentTerminal) {
    // Once the server has reached a terminal state, a delayed non-terminal
    // snapshot can never revive it.  A later terminal event may still carry a
    // genuine progression (for example, a newer authoritative completion).
    return incomingTerminal && incomingEvent > currentEvent;
  }
  return incomingTerminal || incomingEvent > currentEvent;
}

async function persistOnce(
  record: RecipeExecutionRecord,
  key: string,
  subject: string,
): Promise<void> {
  const metadata = serializeExecutionMetadata(record);
  const knownCurrent = persistedExecutions.get(key);
  if (knownCurrent && TERMINAL.has(knownCurrent.status)) {
    const incomingTerminal = TERMINAL.has(metadata.status);
    const incomingEvent = metadata.lastEventId ?? -1;
    const currentEvent = knownCurrent.lastEventId ?? -1;
    if (
      !incomingTerminal ||
      incomingEvent < currentEvent ||
      (metadata.status !== knownCurrent.status &&
        incomingEvent === currentEvent)
    ) {
      return;
    }
  }
  let expectedRevision = revisions.get(key) ?? record.revision;
  let lastConflict: UserAssetApiError | null = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      // A debounced write may wake after logout/account switch. Check the
      // captured owner at the last possible point before authFetch reads the
      // current token, and again before accepting the response.
      assertSubjectUnchanged(subject);
      const saved = await upsertServerRecipeExecution<PersistedRecipeExecution>(
        {
          recipeId: record.recipeId,
          executionId: record.id,
          metadata,
          revision: expectedRevision,
        },
      );
      assertSubjectUnchanged(subject);
      revisions.set(key, saved.revision);
      persistedExecutions.set(key, saved);
      return;
    } catch (error) {
      assertSubjectUnchanged(subject);
      if (!(error instanceof UserAssetApiError) || error.status !== 409)
        throw error;
      const current = error.detail.current as
        | PersistedRecipeExecution
        | undefined;
      if (!current) throw error;
      revisions.set(key, current.revision);
      persistedExecutions.set(key, current);
      if (!incomingWins(metadata, current)) return;
      expectedRevision = current.revision;
      lastConflict = error;
    }
  }
  throw lastConflict ?? new Error("Execution write did not converge.");
}

export async function listRecipeExecutionPage(
  recipeId: string,
  options: { cursor?: string | null; limit?: number } = {},
): Promise<RecipeExecutionPage<PersistedRecipeExecution>> {
  const subject = syncSubject();
  const page = await listServerRecipeExecutions<PersistedRecipeExecution>(
    recipeId,
    { cursor: options.cursor, limit: options.limit ?? 100 },
  );
  assertSubjectUnchanged(subject);
  for (const execution of page.executions) {
    revisions.set(executionKey(subject, execution.id), execution.revision);
    persistedExecutions.set(executionKey(subject, execution.id), execution);
  }
  if (page.resumable) {
    revisions.set(
      executionKey(subject, page.resumable.id),
      page.resumable.revision,
    );
    persistedExecutions.set(
      executionKey(subject, page.resumable.id),
      page.resumable,
    );
  }
  return page;
}

export async function listRecipeExecutions(
  recipeId: string,
): Promise<PersistedRecipeExecution[]> {
  const executions: PersistedRecipeExecution[] = [];
  const executionIds = new Set<string>();
  const seenCursors = new Set<string>();
  let cursor: string | null | undefined;
  let resumable: PersistedRecipeExecution | null = null;

  do {
    const page = await listRecipeExecutionPage(recipeId, { cursor });
    for (const execution of page.executions) {
      if (!executionIds.has(execution.id)) {
        executions.push(execution);
        executionIds.add(execution.id);
      }
    }
    resumable ??= page.resumable ?? null;
    if (page.nextCursor && seenCursors.has(page.nextCursor)) {
      throw new Error(
        "Execution history pagination returned a repeated cursor.",
      );
    }
    if (page.nextCursor) seenCursors.add(page.nextCursor);
    cursor = page.nextCursor;
  } while (cursor);

  if (resumable && !executionIds.has(resumable.id)) {
    executions.push(resumable);
  }
  return executions;
}

async function drainWrites(key: string, state: WriteState): Promise<void> {
  if (state.running) return;
  state.running = true;
  state.timer = null;
  try {
    while (state.latest) {
      const next = state.latest;
      state.latest = null;
      assertSubjectUnchanged(state.subject);
      await persistOnce(next, key, state.subject);
    }
    for (const waiter of state.waiters.splice(0)) waiter.resolve();
  } catch (error) {
    for (const waiter of state.waiters.splice(0)) waiter.reject(error);
  } finally {
    state.running = false;
    if (state.latest) {
      void drainWrites(key, state);
    } else {
      writeStates.delete(key);
    }
  }
}

export function saveRecipeExecution(
  execution: RecipeExecutionRecord,
): Promise<void> {
  const subject = syncSubject();
  const key = executionKey(subject, execution.id);
  const state = writeStates.get(key) ?? {
    subject,
    latest: null,
    running: false,
    timer: null,
    waiters: [],
  };
  state.latest = execution;
  writeStates.set(key, state);
  const promise = new Promise<void>((resolve, reject) => {
    state.waiters.push({ resolve, reject });
  });
  const terminal = TERMINAL.has(execution.status);
  if (terminal && state.timer) {
    clearTimeout(state.timer);
    state.timer = null;
  }
  if (!state.running && !state.timer) {
    if (terminal) void drainWrites(key, state);
    else state.timer = setTimeout(() => void drainWrites(key, state), 200);
  }
  return promise;
}
