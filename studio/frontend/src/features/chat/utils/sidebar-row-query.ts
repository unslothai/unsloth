// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SidebarRowsRequest } from "../api/chat-api";

export const SIDEBAR_PAGE_SIZE = 20;

export type SidebarRowQuery = Omit<SidebarRowsRequest, "limit" | "cursor">;

export function sidebarRowQuery(
  query: Partial<SidebarRowQuery>,
): SidebarRowQuery {
  return {
    archived: query.archived ?? false,
    projectId: query.projectId ?? null,
    unfiledOnly: query.unfiledOnly ?? false,
    includeIds: query.includeIds ?? null,
    excludeIds: [...new Set(query.excludeIds ?? [])].sort(),
    sort: query.sort ?? "updated",
    manualOrder: query.sort === "manual" ? (query.manualOrder ?? []) : null,
  };
}

export function sidebarRowQueryKey(query: SidebarRowQuery): string {
  return JSON.stringify(query);
}

interface QueryRow {
  id: string;
  projectId?: string | null;
  threadIds?: string[];
}

/** The server's filter and order on the client, for a server without the paged route. */
export function applySidebarRowQuery<T extends QueryRow>(
  rows: readonly T[],
  query: SidebarRowQuery,
): T[] {
  const include = query.includeIds ? new Set(query.includeIds) : null;
  const exclude = new Set(query.excludeIds);
  const out = rows.filter((row) => {
    if (include && !include.has(row.id) && !row.threadIds?.some((id) => include.has(id))) {
      return false;
    }
    if (exclude.has(row.id)) return false;
    if (
      query.projectId !== null &&
      (row.projectId ?? null) !== query.projectId
    ) {
      return false;
    }
    if (query.unfiledOnly && row.projectId) return false;
    return true;
  });
  if (query.sort !== "manual" || !query.manualOrder?.length) return out;
  const rank = new Map(query.manualOrder.map((id, index) => [id, index]));
  return out.sort((a, b) => (rank.get(a.id) ?? -1) - (rank.get(b.id) ?? -1));
}

/** `next` orders the loaded rows; rows of `full` it omits keep their places after them. */
export function mergeManualOrderTail(
  next: readonly string[],
  full: readonly string[],
): string[] {
  const seen = new Set(next);
  const out = [...next];
  for (const id of full) {
    if (!seen.has(id)) {
      seen.add(id);
      out.push(id);
    }
  }
  return out;
}
