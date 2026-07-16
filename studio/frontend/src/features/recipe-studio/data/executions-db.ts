// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSubjectKey } from "@/features/auth";
import {
  type RecipeExecutionPage,
  UserAssetApiError,
  listServerRecipeExecutions,
  upsertServerRecipeExecution,
} from "@/features/user-assets";
// Persistence policy is infrastructure, not a feature UI dependency.
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
  owner: RecipeExecutionPersistenceOwner;
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

export type RecipeExecutionPersistenceOwner = {
  subjectKey: string;
  recipeId: string;
  generation: number;
  isCurrent: () => boolean;
};

function syncSubject(expectedSubject?: string): string {
  const subject = getAuthSubjectKey();
  if (expectedSubject && subject !== expectedSubject) {
    throw new Error("Execution persistence account changed.");
  }
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

function assertOwnerCurrent(
  owner: RecipeExecutionPersistenceOwner,
  record?: RecipeExecutionRecord,
): void {
  assertSubjectUnchanged(owner.subjectKey);
  if (!owner.isCurrent()) {
    throw new Error("Execution persistence owner changed.");
  }
  if (record && record.recipeId !== owner.recipeId) {
    throw new Error("Execution persistence recipe changed.");
  }
}

function executionKey(subject: string, recipeId: string, id: string): string {
  return `${subject}\u0000${recipeId}\u0000${id}`;
}

function executionWriteKey(
  owner: RecipeExecutionPersistenceOwner,
  id: string,
): string {
  return `${executionKey(owner.subjectKey, owner.recipeId, id)}\u0000${owner.generation}`;
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

/** Only this projection crosses the persistence boundary. */
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

  // Keep durable resume fields first.
  // If they exceed the cap, persist the largest deterministic
  // completed-column prefix instead of a rejected payload.
  metadata = fitCompletedColumns(metadata);

  // Add snapshots in display priority; omit any that exceeds 256 KiB.
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

function preferString(
  current: string | null,
  incoming: string | null,
): string | null {
  return incoming || current;
}

function mergeStringLists(current: string[], incoming: string[]): string[] {
  const merged = [...current];
  const seen = new Set(current);
  for (const item of incoming) {
    if (!seen.has(item)) {
      seen.add(item);
      merged.push(item);
    }
  }
  return merged;
}

function mergeJsonValue(current: unknown, incoming: unknown): unknown {
  if (incoming === null || incoming === undefined) return current;
  if (current === null || current === undefined) return incoming;
  if (Array.isArray(current) && Array.isArray(incoming)) {
    return incoming.length > current.length ? incoming : current;
  }
  if (
    typeof current === "object" &&
    !Array.isArray(current) &&
    typeof incoming === "object" &&
    !Array.isArray(incoming)
  ) {
    const merged = { ...(current as Record<string, unknown>) };
    for (const [key, value] of Object.entries(
      incoming as Record<string, unknown>,
    )) {
      merged[key] = mergeJsonValue(merged[key], value);
    }
    return merged;
  }
  return incoming;
}

function mergeOptionalObject<T extends object>(
  current: T | null,
  incoming: T | null,
): T | null {
  return mergeJsonValue(current, incoming) as T | null;
}

function persistedMetadata(
  current: PersistedRecipeExecution,
): RecipeExecutionMetadata {
  const metadata = { ...current };
  Reflect.deleteProperty(metadata, "revision");
  Reflect.deleteProperty(metadata, "updatedAt");
  return metadata;
}

function mergeSameTerminalSnapshot(
  incoming: RecipeExecutionMetadata,
  current: PersistedRecipeExecution,
): RecipeExecutionMetadata | null {
  const currentMetadata = persistedMetadata(current);
  const merged: RecipeExecutionMetadata = {
    ...currentMetadata,
    jobId: preferString(current.jobId, incoming.jobId),
    run_name: preferString(current.run_name, incoming.run_name),
    rows: Math.max(current.rows, incoming.rows),
    recipeSignature:
      preferString(current.recipeSignature, incoming.recipeSignature) ?? "",
    stage: preferString(current.stage, incoming.stage),
    current_column: preferString(
      current.current_column,
      incoming.current_column,
    ),
    completed_columns: mergeStringLists(
      current.completed_columns,
      incoming.completed_columns,
    ),
    progress: mergeOptionalObject(current.progress, incoming.progress),
    column_progress: mergeOptionalObject(
      current.column_progress,
      incoming.column_progress,
    ),
    batch: mergeOptionalObject(current.batch, incoming.batch),
    source_progress: mergeOptionalObject(
      current.source_progress,
      incoming.source_progress,
    ),
    model_usage: mergeOptionalObject(
      current.model_usage,
      incoming.model_usage,
    ),
    datasetTotal: Math.max(current.datasetTotal, incoming.datasetTotal),
    analysis: mergeOptionalObject(current.analysis, incoming.analysis),
    error: preferString(current.error, incoming.error),
    createdAt: Math.min(current.createdAt, incoming.createdAt),
    finishedAt:
      current.finishedAt === null
        ? incoming.finishedAt
        : incoming.finishedAt === null
          ? current.finishedAt
          : Math.max(current.finishedAt, incoming.finishedAt),
  };
  if (!fitsExecutionBudget(merged)) return null;
  return JSON.stringify(merged) === JSON.stringify(currentMetadata)
    ? null
    : merged;
}

function reconcileIncoming(
  incoming: RecipeExecutionMetadata,
  current: PersistedRecipeExecution,
): RecipeExecutionMetadata | null {
  const incomingEvent = incoming.lastEventId ?? -1;
  const currentEvent = current.lastEventId ?? -1;
  const incomingTerminal = TERMINAL.has(incoming.status);
  const currentTerminal = TERMINAL.has(current.status);
  if (currentTerminal) {
    // Terminal state rejects stale nonterminal updates; only newer terminal events advance.
    if (!incomingTerminal) return null;
    if (incomingEvent !== currentEvent) {
      return incomingEvent > currentEvent ? incoming : null;
    }
    if (incoming.status !== current.status) return null;

    // Same-event terminal writers merge enrichment without dropping stored fields.
    return mergeSameTerminalSnapshot(incoming, current);
  }
  return incomingTerminal || incomingEvent > currentEvent ? incoming : null;
}

async function persistOnce(
  record: RecipeExecutionRecord,
  key: string,
  owner: RecipeExecutionPersistenceOwner,
): Promise<void> {
  assertOwnerCurrent(owner, record);
  let metadata = serializeExecutionMetadata(record);
  const knownCurrent = persistedExecutions.get(key);
  if (knownCurrent && TERMINAL.has(knownCurrent.status)) {
    const reconciled = reconcileIncoming(metadata, knownCurrent);
    if (!reconciled) return;
    metadata = reconciled;
  }
  let expectedRevision = revisions.get(key) ?? record.revision;
  let lastConflict: UserAssetApiError | null = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      // Debounced writes recheck ownership before sending and after receiving.
      assertOwnerCurrent(owner, record);
      const saved = await upsertServerRecipeExecution<PersistedRecipeExecution>(
        {
          recipeId: record.recipeId,
          executionId: record.id,
          metadata,
          revision: expectedRevision,
        },
        { expectedSubjectKey: owner.subjectKey },
      );
      assertOwnerCurrent(owner, record);
      revisions.set(key, saved.revision);
      persistedExecutions.set(key, saved);
      return;
    } catch (error) {
      assertOwnerCurrent(owner, record);
      if (!(error instanceof UserAssetApiError) || error.status !== 409)
        throw error;
      const current = error.detail.current as
        | PersistedRecipeExecution
        | undefined;
      if (!current) throw error;
      revisions.set(key, current.revision);
      persistedExecutions.set(key, current);
      const reconciled = reconcileIncoming(metadata, current);
      if (!reconciled) return;
      metadata = reconciled;
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
    revisions.set(
      executionKey(subject, recipeId, execution.id),
      execution.revision,
    );
    persistedExecutions.set(
      executionKey(subject, recipeId, execution.id),
      execution,
    );
  }
  if (page.resumable) {
    revisions.set(
      executionKey(subject, recipeId, page.resumable.id),
      page.resumable.revision,
    );
    persistedExecutions.set(
      executionKey(subject, recipeId, page.resumable.id),
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
      assertOwnerCurrent(state.owner, next);
      const revisionKey = executionKey(
        state.owner.subjectKey,
        state.owner.recipeId,
        next.id,
      );
      await persistOnce(next, revisionKey, state.owner);
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
  owner: RecipeExecutionPersistenceOwner,
): Promise<void> {
  syncSubject(owner.subjectKey);
  assertOwnerCurrent(owner, execution);
  const key = executionWriteKey(owner, execution.id);
  const state = writeStates.get(key) ?? {
    owner,
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
