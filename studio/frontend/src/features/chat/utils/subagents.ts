// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure helpers for the subagent panel: polling cadence, list merging, display strings.

import type { SubagentRecord } from "../types/subagents.ts";

/** The model-facing tool the client enables through `enabled_tools`. */
export const SUBAGENT_TOOL_NAME = "spawn_subagent";

export const SUBAGENT_LIST_POLL_ACTIVE_MS = 1500;
export const SUBAGENT_LIST_POLL_IDLE_MS = 10_000;
export const SUBAGENT_DETAIL_POLL_MS = 1000;
/** An older backend answers 404 for the whole route; asking again every second only adds noise. */
export const SUBAGENT_LIST_POLL_UNAVAILABLE_MS = 60_000;

export function isSubagentRunning(
  record: Pick<SubagentRecord, "status">,
): boolean {
  return record.status === "running";
}

export function anySubagentRunning(
  records: readonly Pick<SubagentRecord, "status">[] | undefined,
): boolean {
  return records?.some(isSubagentRunning) ?? false;
}

/** Fast while a subagent runs or the parent turn can still spawn one, slow otherwise. */
export function subagentListPollInterval(input: {
  anyRunning: boolean;
  threadGenerating: boolean;
}): number {
  return input.anyRunning || input.threadGenerating
    ? SUBAGENT_LIST_POLL_ACTIVE_MS
    : SUBAGENT_LIST_POLL_IDLE_MS;
}

function sameJson(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

/** Fields a list row renders, so an unchanged poll keeps the old object and skips a re-render. */
export function sameSubagentSummary(
  a: SubagentRecord,
  b: SubagentRecord,
): boolean {
  return (
    a.id === b.id &&
    a.childThreadId === b.childThreadId &&
    a.sessionId === b.sessionId &&
    a.description === b.description &&
    a.model === b.model &&
    a.status === b.status &&
    a.error === b.error &&
    a.resultPreview === b.resultPreview &&
    a.toolCallCount === b.toolCallCount &&
    a.startedAt === b.startedAt &&
    a.finishedAt === b.finishedAt &&
    sameJson(a.pendingApproval, b.pendingApproval)
  );
}

/**
 * The server's list replaces the local one, oldest first. Rows that did not change keep their
 * object, and the returned array is `prev` itself when nothing changed at all.
 */
export function mergeSubagentList(
  prev: readonly SubagentRecord[] | undefined,
  next: readonly SubagentRecord[],
): readonly SubagentRecord[] {
  const before = new Map((prev ?? []).map((record) => [record.id, record]));
  const merged = [...next]
    .sort((a, b) => a.startedAt - b.startedAt)
    .map((record) => {
      const old = before.get(record.id);
      return old && sameSubagentSummary(old, record) ? old : record;
    });
  if (
    prev &&
    prev.length === merged.length &&
    merged.every((record, index) => record === prev[index])
  ) {
    return prev;
  }
  return merged;
}

/**
 * A detail read replaces the cached record only when it carries something new, so a poll of a
 * finished subagent does not re-render its transcript.
 */
export function mergeSubagentDetail(
  prev: SubagentRecord | undefined,
  next: SubagentRecord,
): SubagentRecord {
  if (
    prev &&
    sameSubagentSummary(prev, next) &&
    sameJson(prev.transcript, next.transcript)
  ) {
    return prev;
  }
  return next;
}

/**
 * Opens the panel when a thread gets its first subagent. `prevCount` is what the previous poll
 * saw: undefined for a thread this session never polled, so an old chat that already has
 * subagents stays closed unless one of them is still running.
 */
export function shouldAutoOpenSubagentPanel(input: {
  prevCount: number | undefined;
  next: readonly Pick<SubagentRecord, "status">[];
  alreadyAutoOpened: boolean;
}): boolean {
  if (input.alreadyAutoOpened || input.next.length === 0) return false;
  if (input.prevCount === undefined) return anySubagentRunning(input.next);
  return input.prevCount === 0;
}

export function subagentElapsedMs(
  record: Pick<SubagentRecord, "startedAt" | "finishedAt" | "status">,
  now: number,
): number {
  const end =
    record.finishedAt ?? (isSubagentRunning(record) ? now : record.startedAt);
  return Math.max(0, end - record.startedAt);
}

export function formatSubagentDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

/** "openai/gpt-5.2" becomes "gpt-5.2": the vendor prefix only takes room in a narrow row. */
export function shortSubagentModel(model: string | null | undefined): string {
  if (!model) return "";
  const trimmed = model.trim();
  const slash = trimmed.lastIndexOf("/");
  return slash >= 0 && slash < trimmed.length - 1
    ? trimmed.slice(slash + 1)
    : trimmed;
}

/** One line for a tool card: the arguments as key: value pairs, clipped. */
export function summarizeSubagentArguments(
  args: Record<string, unknown> | null | undefined,
  maxLength = 120,
): string {
  if (!args || typeof args !== "object") return "";
  const parts: string[] = [];
  for (const [key, value] of Object.entries(args)) {
    if (value == null) continue;
    const text =
      typeof value === "string"
        ? value
        : typeof value === "number" || typeof value === "boolean"
          ? String(value)
          : JSON.stringify(value);
    if (!text) continue;
    parts.push(`${key}: ${text.replace(/\s+/g, " ")}`);
  }
  const joined = parts.join(", ");
  return joined.length > maxLength
    ? `${joined.slice(0, maxLength - 1)}…`
    : joined;
}

/**
 * The subagent a spawn_subagent tool card belongs to. The card knows only its description, so a
 * description used by several subagents is ambiguous and resolves to none.
 */
export function subagentForDescription(
  records: readonly SubagentRecord[] | undefined,
  description: string,
): SubagentRecord | null {
  const wanted = description.trim();
  if (!wanted) return null;
  const matches = (records ?? []).filter(
    (record) => record.description.trim() === wanted,
  );
  return matches.length === 1 ? matches[0] : null;
}
