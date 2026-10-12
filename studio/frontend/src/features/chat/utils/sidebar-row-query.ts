// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SidebarRowsRequest } from "../api/chat-api";

/** Rows a sidebar list fetches up front and per scroll. */
export const SIDEBAR_PAGE_SIZE = 20;

/** One sidebar list, without its page window. */
export type SidebarRowQuery = Omit<SidebarRowsRequest, "limit" | "cursor">;

export function sidebarRowQuery(
  query: Partial<SidebarRowQuery>,
): SidebarRowQuery {
  return {
    archived: query.archived ?? false,
    projectId: query.projectId ?? null,
    unfiledOnly: query.unfiledOnly ?? false,
    includeIds: query.includeIds ?? null,
    // Sorted so the same set is the same list however the stores happened to order it.
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
}

/** The server's filter and order, run over rows already grouped and newest first. Used against a
 *  server without the paged route, so both paths list the same rows in the same order. */
export function applySidebarRowQuery<T extends QueryRow>(
  rows: readonly T[],
  query: SidebarRowQuery,
): T[] {
  const include = query.includeIds ? new Set(query.includeIds) : null;
  const exclude = new Set(query.excludeIds);
  const out = rows.filter((row) => {
    if (include && !include.has(row.id)) return false;
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
  // applyManualOrder: unranked rows first in incoming order, then ranked rows by rank.
  const rank = new Map(query.manualOrder.map((id, index) => [id, index]));
  return out.sort((a, b) => (rank.get(a.id) ?? -1) - (rank.get(b.id) ?? -1));
}

/** A manual order written from a partly loaded list: `next` orders the loaded rows, and every row
 *  of `full` it does not mention keeps its place after them, so unloaded rows are never dropped. */
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
