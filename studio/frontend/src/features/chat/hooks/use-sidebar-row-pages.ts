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
import { type RowPageResult, RowPager } from "../utils/sidebar-row-pager";
import {
  SIDEBAR_PAGE_SIZE,
  type SidebarRowQuery,
  applySidebarRowQuery,
  sidebarRowQueryKey,
} from "../utils/sidebar-row-query";
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
  const ids: string[] = [];
  let cursor: string | null = null;
  for (let guard = 0; guard < 10_000; guard++) {
    const page: RowPageResult<SidebarItem> = await fetchSidebarRowPage(
      query,
      200,
      cursor,
    );
    for (const row of page.rows) ids.push(row.id);
    if (!page.hasMore || !page.nextCursor || page.rows.length === 0) break;
    cursor = page.nextCursor;
  }
  return ids;
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

/** `items` is every loaded row once, in slot order: what the sidebar derives its lists from. */
export function useSidebarRowPages(
  slots: Record<string, SidebarRowSlot | null>,
  enabled: boolean,
): {
  lists: Record<string, SidebarRowPages>;
  items: SidebarItem[];
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
    const unsubscribes = [...pagers.values()].map((pager) =>
      pager.subscribe(bump),
    );
    bump();
    return () => {
      for (const unsubscribe of unsubscribes) unsubscribe();
    };
  }, [slotKey, enabled]);

  useEffect(() => {
    const pagers = pagersRef.current;
    return () => {
      for (const pager of pagers.values()) unregisterPager(pager);
      pagers.clear();
    };
  }, []);

  const unionRef = useRef<{ parts: SidebarItem[][]; items: SidebarItem[] }>({
    parts: [],
    items: EMPTY_ROWS,
  });

  // biome-ignore lint/correctness/useExhaustiveDependencies: `version` moves on every pager change
  return useMemo(() => {
    const lists: Record<string, SidebarRowPages> = {};
    const parts: SidebarItem[][] = [];
    for (const [name] of slotsRef.current) {
      const pager = pagersRef.current.get(name);
      if (!pager) {
        lists[name] = IDLE_LIST;
        continue;
      }
      const state = pager.getState();
      parts.push(state.rows);
      lists[name] = {
        ...state,
        // nothing renders `error`, so the next scroll is the retry
        loadMore: () => (state.error ? pager.retry() : pager.loadMore()),
        loadAll: () => pager.loadAll(),
        retry: () => pager.retry(),
      };
    }
    // keep identity: every sidebar memo depends on this array
    const previous = unionRef.current;
    if (
      previous.parts.length !== parts.length ||
      previous.parts.some((part, index) => part !== parts[index])
    ) {
      const seen = new Set<string>();
      const items: SidebarItem[] = [];
      for (const part of parts) {
        for (const item of part) {
          if (seen.has(item.id)) continue;
          seen.add(item.id);
          items.push(item);
        }
      }
      unionRef.current = { parts, items: items.length ? items : EMPTY_ROWS };
    }
    return { lists, items: unionRef.current.items };
  }, [version, slotKey]);
}
