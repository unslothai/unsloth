// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSessionEpoch } from "@/features/auth";
import { useEffect, useMemo, useReducer, useRef } from "react";
import {
  CHAT_HISTORY_UPDATED_EVENT,
  type SidebarRowRecord,
  fetchSidebarRows,
} from "../api/chat-api";
import {
  listStoredChatThreads,
  settleLegacyChatImport,
} from "../utils/chat-history-storage";
import { isChatThreadDeleted } from "../utils/chat-thread-tombstones";
import { repairLegacyChatTitles } from "../utils/repair-legacy-chat-titles";
import { type RowPageResult, RowPager } from "../utils/sidebar-row-pager";
import {
  SIDEBAR_PAGE_SIZE,
  type SidebarRowQuery,
  applySidebarRowQuery,
  sidebarRowQueryKey,
} from "../utils/sidebar-row-query";
import type { ThreadRecord } from "../types";
import { type SidebarItem, groupThreads } from "./use-chat-sidebar-items";

const SIDEBAR_REFRESH_DEBOUNCE_MS = 300;
const FALLBACK_CURSOR = "full:";

function toSidebarItem(row: SidebarRowRecord): SidebarItem | null {
  // a delete tombstones its threads before the server forgets them
  const threadIds = (row.threadIds?.length ? row.threadIds : [row.id]).filter(
    (id) => !isChatThreadDeleted(id),
  );
  if (threadIds.length === 0) return null;
  return {
    type: row.type,
    id: row.id,
    threadIds,
    title: row.title,
    createdAt: row.createdAt,
    updatedAt: row.updatedAt,
    ...(row.modifiedAt ? { modifiedAt: row.modifiedAt } : {}),
    ...(row.modelIds?.length ? { modelIds: row.modelIds } : {}),
    ...(row.type === "single" ? { isFork: Boolean(row.isFork) } : {}),
    projectId: row.projectId ?? null,
  };
}

let serverPages: boolean | null = null;

// Without the paged route: one full read per history change, shared by every list.
let fullList: {
  epoch: number;
  session: number;
  promise: Promise<Awaited<ReturnType<typeof listStoredChatThreads>>>;
} | null = null;
let fullListEpoch = 0;

function readFullList() {
  const session = getAuthSessionEpoch();
  if (
    !fullList ||
    fullList.epoch !== fullListEpoch ||
    fullList.session !== session
  ) {
    const promise = listStoredChatThreads({ includeArchived: true });
    fullList = { epoch: fullListEpoch, session, promise };
    void promise.then(repairLegacyChatTitles).catch(() => undefined);
    promise.catch(() => {
      if (fullList?.promise === promise) fullList = null;
    });
  }
  return fullList.promise;
}

async function fullListPage(
  query: SidebarRowQuery,
  limit: number,
  cursor: string | null,
): Promise<RowPageResult<SidebarItem>> {
  const rows = applySidebarRowQuery(
    groupThreads(await readFullList(), query.archived),
    query,
  );
  const offset = cursor?.startsWith(FALLBACK_CURSOR)
    ? Number(cursor.slice(FALLBACK_CURSOR.length)) || 0
    : 0;
  const end = offset + limit;
  const hasMore = end < rows.length;
  return {
    rows: rows.slice(offset, end),
    nextCursor: hasMore ? `${FALLBACK_CURSOR}${end}` : null,
    hasMore,
    total: rows.length,
  };
}

export async function fetchSidebarRowPage(
  query: SidebarRowQuery,
  limit: number,
  cursor: string | null,
): Promise<RowPageResult<SidebarItem>> {
  const continuesFullList = cursor?.startsWith(FALLBACK_CURSOR) ?? false;
  if (
    !continuesFullList &&
    serverPages !== false &&
    (await settleLegacyChatImport())
  ) {
    const page = await fetchSidebarRows({ ...query, limit, cursor });
    if (page) {
      serverPages = true;
      // pre-cut legacy titles are repaired as their rows load, as the full listing did for all
      const shown = page.rows.flatMap((row) =>
        (row.threadIds?.length ? row.threadIds : [row.id]).map((id) => ({ id, title: row.title })),
      );
      void repairLegacyChatTitles(shown as ThreadRecord[]).catch(() => undefined);
      return {
        rows: page.rows
          .map(toSidebarItem)
          .filter((row): row is SidebarItem => row !== null),
        nextCursor: page.nextCursor,
        hasMore: page.hasMore,
        total: page.total ?? null,
      };
    }
    serverPages = false;
  }
  return fullListPage(query, limit, continuesFullList ? cursor : null);
}

/** Every row id in order, so a manual order written from a partly loaded list keeps unloaded rows. */
export async function readAllSidebarRowIds(
  query: SidebarRowQuery,
): Promise<string[]> {
  return (await readAllSidebarRows(query)).map((row) => row.id);
}

/** Every row of a query, so a whole-list action never takes the loaded pages for the list. */
export async function readAllSidebarRows(
  query: SidebarRowQuery,
): Promise<SidebarItem[]> {
  const rows: SidebarItem[] = [];
  let cursor: string | null = null;
  for (let guard = 0; guard < 10_000; guard++) {
    const page: RowPageResult<SidebarItem> = await fetchSidebarRowPage(
      query,
      200,
      cursor,
    );
    rows.push(...page.rows);
    if (!page.hasMore || !page.nextCursor || page.rows.length === 0) break;
    cursor = page.nextCursor;
  }
  return rows;
}

type SidebarPager = RowPager<SidebarRowQuery, SidebarItem>;

const livePagers = new Set<SidebarPager>();
let refreshTimer: ReturnType<typeof setTimeout> | null = null;

function onHistoryUpdated() {
  if (refreshTimer !== null) clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => {
    refreshTimer = null;
    fullListEpoch += 1;
    for (const pager of livePagers) {
      const { loaded, loading } = pager.getState();
      if (loaded || loading) void pager.refresh();
    }
  }, SIDEBAR_REFRESH_DEBOUNCE_MS);
}

function registerPager(pager: SidebarPager) {
  if (livePagers.size === 0 && typeof window !== "undefined") {
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
  }
  livePagers.add(pager);
}

function unregisterPager(pager: SidebarPager) {
  livePagers.delete(pager);
  if (livePagers.size === 0 && typeof window !== "undefined") {
    window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    if (refreshTimer !== null) clearTimeout(refreshTimer);
    refreshTimer = null;
  }
}

export interface SidebarRowSlot {
  query: SidebarRowQuery;
  initialLimit?: number;
  /** loaded for row lookups only: its rows are in `items` but never in `listedItems` */
  lookupOnly?: boolean;
}

export interface SidebarRowPages {
  rows: SidebarItem[];
  hasMore: boolean;
  total: number | null;
  loaded: boolean;
  loading: boolean;
  error: unknown;
  loadMore: () => Promise<void>;
  loadAll: () => Promise<void>;
  retry: () => Promise<void>;
}

const EMPTY_ROWS: SidebarItem[] = [];
const IDLE_LIST: SidebarRowPages = {
  rows: EMPTY_ROWS,
  hasMore: false,
  total: null,
  loaded: false,
  loading: false,
  error: null,
  loadMore: () => Promise.resolve(),
  loadAll: () => Promise.resolve(),
  retry: () => Promise.resolve(),
};

type RowUnion = { parts: SidebarItem[][]; items: SidebarItem[] };

/** Every row of `parts` once, in order; the previous array while no part changed. */
function unionRows(previous: RowUnion, parts: SidebarItem[][]): RowUnion {
  if (
    previous.parts.length === parts.length &&
    previous.parts.every((part, index) => part === parts[index])
  ) {
    return previous;
  }
  const seen = new Set<string>();
  const items: SidebarItem[] = [];
  for (const part of parts) {
    for (const item of part) {
      if (seen.has(item.id)) continue;
      seen.add(item.id);
      items.push(item);
    }
  }
  return { parts, items: items.length ? items : EMPTY_ROWS };
}

/** `items` is every loaded row once, in slot order, for lookups; `listedItems` leaves out
 *  lookup-only slots, so a list never draws a row its own pages have not reached. */
export function useSidebarRowPages(
  slots: Record<string, SidebarRowSlot | null>,
  enabled: boolean,
): {
  lists: Record<string, SidebarRowPages>;
  items: SidebarItem[];
  listedItems: SidebarItem[];
} {
  const pagersRef = useRef(new Map<string, SidebarPager>());
  const [version, bump] = useReducer((n: number) => n + 1, 0);

  const active = Object.entries(slots).filter(
    (entry): entry is [string, SidebarRowSlot] => entry[1] !== null,
  );
  const slotKey = enabled
    ? JSON.stringify(
        active.map(([name, slot]) => [
          name,
          sidebarRowQueryKey(slot.query),
          slot.initialLimit ?? SIDEBAR_PAGE_SIZE,
        ]),
      )
    : "";
  const slotsRef = useRef(active);
  slotsRef.current = active;

  // biome-ignore lint/correctness/useExhaustiveDependencies: `slotKey` names every slot's query
  useEffect(() => {
    const pagers = pagersRef.current;
    const wanted = enabled ? slotsRef.current : [];
    const names = new Set(wanted.map(([name]) => name));
    for (const [name, pager] of pagers) {
      if (names.has(name)) continue;
      unregisterPager(pager);
      pagers.delete(name);
    }
    for (const [name, slot] of wanted) {
      const key = sidebarRowQueryKey(slot.query);
      const existing = pagers.get(name);
      if (existing) {
        void existing.setQuery(slot.query, key);
        continue;
      }
      const pager: SidebarPager = new RowPager(
        fetchSidebarRowPage,
        slot.query,
        key,
        {
          initialLimit: slot.initialLimit ?? SIDEBAR_PAGE_SIZE,
          pageSize: SIDEBAR_PAGE_SIZE,
        },
      );
      pagers.set(name, pager);
      registerPager(pager);
      void pager.start();
    }
    // one sidebar render per 16 ms however many lists answer; a timer, as a page drawing no frames never runs rAF
    let timer: ReturnType<typeof setTimeout> | null = null;
    const schedule = () => {
      if (timer !== null) return;
      timer = setTimeout(() => {
        timer = null;
        bump();
      }, 16);
    };
    const unsubscribes = [...pagers.values()].map((pager) =>
      pager.subscribe(schedule),
    );
    bump();
    return () => {
      for (const unsubscribe of unsubscribes) unsubscribe();
      if (timer !== null) clearTimeout(timer);
    };
  }, [slotKey, enabled]);

  useEffect(() => {
    const pagers = pagersRef.current;
    return () => {
      for (const pager of pagers.values()) unregisterPager(pager);
      pagers.clear();
    };
  }, []);

  const unionRef = useRef<RowUnion>({ parts: [], items: EMPTY_ROWS });
  const listedRef = useRef<RowUnion>({ parts: [], items: EMPTY_ROWS });

  // biome-ignore lint/correctness/useExhaustiveDependencies: `version` moves on every pager change
  return useMemo(() => {
    const lists: Record<string, SidebarRowPages> = {};
    const parts: SidebarItem[][] = [];
    const listedParts: SidebarItem[][] = [];
    for (const [name, slot] of slotsRef.current) {
      const pager = pagersRef.current.get(name);
      if (!pager) {
        lists[name] = IDLE_LIST;
        continue;
      }
      const state = pager.getState();
      parts.push(state.rows);
      if (!slot.lookupOnly) listedParts.push(state.rows);
      lists[name] = {
        ...state,
        // nothing renders `error`, so the next scroll is the retry
        loadMore: () => (state.error ? pager.retry() : pager.loadMore()),
        // rejects when a page failed, so a caller never takes the partial list for the whole
        loadAll: async () => {
          await pager.loadAll();
          // render now, not on the batched timer: a caller acting on the rows (Select All) renders next
          bump();
          const { error } = pager.getState();
          if (error) throw error;
        },
        retry: () => pager.retry(),
      };
    }
    // keep identity: every sidebar memo depends on these arrays
    unionRef.current = unionRows(unionRef.current, parts);
    listedRef.current = unionRows(listedRef.current, listedParts);
    return { lists, items: unionRef.current.items, listedItems: listedRef.current.items };
  }, [version, slotKey]);
}
