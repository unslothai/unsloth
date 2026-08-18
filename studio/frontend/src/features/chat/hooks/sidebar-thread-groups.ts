// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sidebar's thread grouping, kept apart from useChatSidebarItems so it can be exercised
// without dragging in chat storage, the runtime store and the artifact store. `groupThreads` is
// unchanged and is still re-exported from ./use-chat-sidebar-items, which is where every existing
// caller imports it from.

import { useMemo } from "react";
import type { ThreadRecord } from "../types";

export interface SidebarItem {
  type: "single" | "compare";
  id: string;
  /** The pane threads behind this row id; `runningByThreadId` is keyed per pane thread. */
  threadIds?: string[];
  title: string;
  createdAt: number;
  updatedAt: number;
  isFork?: boolean;
  projectId?: string | null;
}

function lastActivityAt(thread: ThreadRecord): number {
  return thread.updatedAt ?? thread.createdAt;
}

export function groupThreads(
  threads: ThreadRecord[],
  archived = false,
): SidebarItem[] {
  const items: SidebarItem[] = [];
  const pairItems = new Map<string, SidebarItem>();

  for (const t of threads) {
    // Coerce archived to a boolean before comparing. Legacy threads (from the
    // older browser-only Unsloth, or any record predating the archived field)
    // can have archived === undefined or null; a raw `!== archived` comparison
    // would drop those from BOTH the Recents (archived=false) and Archived
    // (archived=true) lists, hiding existing chats. Treat missing as false.
    if (Boolean(t.archived) !== archived) {
      continue;
    }
    if (t.pairId) {
      const existing = pairItems.get(t.pairId);
      if (existing) {
        existing.createdAt = Math.max(existing.createdAt, t.createdAt);
        existing.updatedAt = Math.max(existing.updatedAt, lastActivityAt(t));
        existing.threadIds?.push(t.id);
        continue;
      }
      const item: SidebarItem = {
        type: "compare",
        id: t.pairId,
        threadIds: [t.id],
        title: t.title,
        createdAt: t.createdAt,
        updatedAt: lastActivityAt(t),
        projectId: t.projectId ?? null,
      };
      pairItems.set(t.pairId, item);
      items.push(item);
    } else if (!t.pairId) {
      items.push({
        type: "single",
        id: t.id,
        threadIds: [t.id],
        title: t.title,
        createdAt: t.createdAt,
        updatedAt: lastActivityAt(t),
        isFork: Boolean(t.forkedFromThreadId),
        projectId: t.projectId ?? null,
      });
    }
  }

  return items.sort((a, b) => b.updatedAt - a.updatedAt);
}

/**
 * Every ThreadRecord field `groupThreads` reads, and nothing else.
 *
 * This list is the memo key, so it has to stay exactly the input set of `groupThreads`. A field
 * added here that groupThreads ignores only costs a wasted regroup; a field groupThreads starts
 * reading that is NOT here would let a changed thread keep the previous grouping, i.e. stale rows
 * in the rail. tests/sidebar-thread-groups-memo.test.ts pins that direction: it feeds records that
 * differ in every OTHER field and asserts the grouping is genuinely identical whenever
 * `sameSidebarThreads` says so.
 */
export const SIDEBAR_THREAD_FIELDS = [
  "id",
  "title",
  "archived",
  "pairId",
  "projectId",
  "createdAt",
  "updatedAt",
  "forkedFromThreadId",
] as const satisfies readonly (keyof ThreadRecord)[];

/**
 * Do two thread lists group into the same sidebar rows?
 *
 * Compared position by position with Object.is over SIDEBAR_THREAD_FIELDS. Deliberately stricter
 * than groupThreads itself, which only reads `archived` and `forkedFromThreadId` through
 * Boolean(): undefined and false read as different here, which can only cost a regroup that was
 * not needed, never skip one that was.
 *
 * Order matters and is not normalised away. groupThreads walks the list in order and its sort is
 * not total when two threads share an updatedAt, so two differently ordered lists with the same
 * members can legitimately group differently.
 */
export function sameSidebarThreads(
  a: readonly ThreadRecord[],
  b: readonly ThreadRecord[],
): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    const left = a[i] as ThreadRecord;
    const right = b[i] as ThreadRecord;
    if (left === right) continue;
    for (const field of SIDEBAR_THREAD_FIELDS) {
      if (!Object.is(left[field], right[field])) return false;
    }
  }
  return true;
}

// One shared empty list, so a hook called with no threads yet does not hand a fresh [] to the
// memo below on every render and defeat it.
const NO_THREADS: ThreadRecord[] = [];

/**
 * The sidebar's two grouped lists, with a stable identity while the threads have not changed.
 *
 * Unmemoized, `groupThreads` allocated a fresh array of fresh objects on every render of every
 * consumer, at O(N log N) each. That is its own cost, and it also broke every downstream useMemo
 * in app-sidebar.tsx keyed on the result (recentChatItems, pinnedChatItems, chatsByProjectId,
 * sidebarProjectRecords, sortedRecentChatItems, recentRowIds, runningChatCount), so one useState
 * flip in AppSidebar re-ran two sorts and seven derivations over the whole history.
 */
export function useSidebarThreadGroups(threads: ThreadRecord[] | undefined): {
  items: SidebarItem[];
  archivedItems: SidebarItem[];
} {
  const source = threads ?? NO_THREADS;
  const items = useMemo(() => groupThreads(source), [source]);
  const archivedItems = useMemo(() => groupThreads(source, true), [source]);
  return { items, archivedItems };
}
