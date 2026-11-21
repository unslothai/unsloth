// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// React-free so the node --test suite can import it.

import type {
  ApiMonitorEntry,
  ApiMonitorResponse,
} from "@/features/chat/types/api";

export type MonitorStatusFilter =
  | "all"
  | "running"
  | "completed"
  | "error"
  | "cancelled";

function sameValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (
    a == null ||
    b == null ||
    typeof a !== "object" ||
    typeof b !== "object"
  ) {
    return false;
  }
  return JSON.stringify(a) === JSON.stringify(b);
}

function sameFields(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
  skip?: string,
): boolean {
  const keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length) return false;
  return keys.every((key) => key === skip || sameValue(a[key], b[key]));
}

function mergeEntries(
  previous: ApiMonitorEntry[],
  next: ApiMonitorEntry[],
): ApiMonitorEntry[] {
  const byId = new Map(previous.map((entry) => [entry.id, entry]));
  let unchanged = previous.length === next.length;
  const merged = next.map((entry, index) => {
    const kept = byId.get(entry.id);
    if (
      kept &&
      sameFields(
        kept as unknown as Record<string, unknown>,
        entry as unknown as Record<string, unknown>,
      )
    ) {
      if (previous[index] !== kept) unchanged = false;
      return kept;
    }
    unchanged = false;
    return entry;
  });
  return unchanged ? previous : merged;
}

export function mergeMonitorResponse(
  previous: ApiMonitorResponse | null,
  next: ApiMonitorResponse,
): ApiMonitorResponse {
  if (previous == null) return next;
  const entries = mergeEntries(previous.entries, next.entries);
  const merged = { ...next, entries };
  if (
    entries === previous.entries &&
    sameFields(
      previous as unknown as Record<string, unknown>,
      merged as unknown as Record<string, unknown>,
      "server_time",
    )
  ) {
    return previous;
  }
  return merged;
}

export function matchesStatus(
  entry: ApiMonitorEntry,
  status: MonitorStatusFilter,
): boolean {
  if (status === "all") return true;
  if (entry.status !== status) return false;
  return entry.kind !== "lifecycle" || status === "error";
}

export function filterEntries(
  entries: ApiMonitorEntry[],
  status: MonitorStatusFilter,
  query: string,
): ApiMonitorEntry[] {
  const needle = query.trim().toLowerCase();
  return entries.filter((entry) => {
    if (!matchesStatus(entry, status)) {
      return false;
    }
    if (!needle) {
      return true;
    }
    // Coerced: one malformed network entry throwing here would blank the whole log.
    return [
      entry.model,
      entry.endpoint,
      entry.prompt_preview,
      entry.reply_preview,
      entry.error,
    ].some((field) =>
      String(field ?? "")
        .toLowerCase()
        .includes(needle),
    );
  });
}

const COUNTED_STATUSES = [
  "running",
  "completed",
  "error",
  "cancelled",
] as const;

export function countByStatus(
  entries: ApiMonitorEntry[],
): Record<MonitorStatusFilter, number> {
  const counts: Record<MonitorStatusFilter, number> = {
    all: entries.length,
    running: 0,
    completed: 0,
    error: 0,
    cancelled: 0,
  };
  for (const entry of entries) {
    for (const status of COUNTED_STATUSES) {
      if (matchesStatus(entry, status)) counts[status] += 1;
    }
  }
  return counts;
}
