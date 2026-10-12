// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as RowPagesModule from "../src/features/chat/hooks/use-sidebar-row-pages.ts";
import { RowPager } from "../src/features/chat/utils/sidebar-row-pager.ts";
import * as rowQuery from "../src/features/chat/utils/sidebar-row-query.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Row = { id: string; projectId?: string | null; updatedAt: number };
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function offsetPage<T>(list: T[], limit: number, cursor: string | null) {
  const offset = cursor ? Number(cursor) : 0;
  const end = offset + limit;
  return {
    rows: list.slice(offset, end),
    nextCursor: end < list.length ? String(end) : null,
    hasMore: end < list.length,
    total: list.length,
  };
}

test("a retry after a failed query change re-reads the new query from the top", async () => {
  const lists: Record<string, Row[]> = {
    a: Array.from({ length: 100 }, (_, i) => ({ id: `a${i}`, updatedAt: 0 })),
    b: Array.from({ length: 100 }, (_, i) => ({ id: `b${i}`, updatedAt: 0 })),
  };
  let failNext = false;
  const pager = new RowPager<string, Row>(
    async (query, limit, cursor) => {
      if (failNext) {
        failNext = false;
        throw new Error("offline");
      }
      return offsetPage(lists[query], limit, cursor);
    },
    "a",
    "a",
    { initialLimit: 20, pageSize: 20 },
  );
  await pager.start();
  failNext = true;
  await pager.setQuery("b", "b");
  assert.ok(pager.getState().error);
  await pager.retry();
  const ids = pager.getState().rows.map((row) => row.id);
  assert.equal(ids[0], "b0");
  assert.ok(ids.every((id) => id.startsWith("b")), "rows mix the old and new query");
});

test("a read replaced by a refresh stops asking the server", async () => {
  const all = Array.from({ length: 2000 }, (_, i) => ({ id: `c${i}`, updatedAt: 0 }));
  const calls: (string | null)[] = [];
  const pager = new RowPager<unknown, Row>(
    async (_query, limit, cursor) => {
      calls.push(cursor);
      await tick();
      return offsetPage(all, limit, cursor);
    },
    {},
    "q",
    { initialLimit: 20, pageSize: 20 },
  );
  await pager.start();
  calls.length = 0;
  const selectAll = pager.loadAll();
  await tick();
  void pager.refresh();
  await selectAll;
  // one walk of 2,000 rows is 10 requests of 200, plus the refresh
  assert.ok(calls.length <= 12, `the list was walked twice: ${calls.length} requests`);
});

/** The hook under a minimal React: state cells by call order, effects re-run on changed deps. */
function hookHarness(fetchSidebarRows: (request: { limit: number; cursor: string | null }) => unknown) {
  (globalThis as { window?: unknown }).window = new EventTarget();
  const cells: { v: unknown }[] = [];
  let at = 0;
  const cell = <T>(init: () => T): { v: T } => {
    const i = at++;
    if (!(i in cells)) cells[i] = { v: init() };
    return cells[i] as { v: T };
  };
  const react = {
    useRef: (value: unknown) => cell(() => ({ current: value })).v,
    useReducer: () => [0, () => {}],
    useMemo: (fn: () => unknown) => fn(),
    useEffect: (fn: () => unknown, deps?: unknown[]) => {
      const c = cell<{ deps?: unknown[] } | null>(() => null);
      const same =
        c.v?.deps && deps && deps.every((dep, k) => Object.is(dep, c.v?.deps?.[k]));
      if (!same) {
        c.v = { deps };
        fn();
      }
    },
  };
  const mod = loadWithStubs<typeof RowPagesModule>(
    new URL("../src/features/chat/hooks/use-sidebar-row-pages.ts", import.meta.url),
    {
      react,
      "@/features/auth": { getAuthSessionEpoch: () => 0 },
      "../api/chat-api": { CHAT_HISTORY_UPDATED_EVENT: "history", fetchSidebarRows },
      "../utils/chat-history-storage": {
        listStoredChatThreads: async () => [],
        settleLegacyChatImport: async () => true,
      },
      "../utils/repair-legacy-chat-titles": { repairLegacyChatTitles: async () => 0 },
      "../utils/chat-thread-tombstones": { isChatThreadDeleted: () => false },
      "../utils/sidebar-row-pager": { RowPager },
      "../utils/sidebar-row-query": rowQuery,
      "./use-chat-sidebar-items": { groupThreads: () => [] },
    },
  );
  const slots = { recents: { query: rowQuery.sidebarRowQuery({}) } };
  const render = () => {
    at = 0;
    return mod.useSidebarRowPages(slots as never, true).lists.recents;
  };
  return { render };
}

function sidebarRows(ids: string[]) {
  return ids.map((id) => ({
    id,
    type: "single",
    threadIds: [id],
    title: "",
    createdAt: 0,
    updatedAt: 0,
  }));
}

test("a history change during a list's first load is not lost", async () => {
  let server = ["old"];
  let release = () => {};
  let gate: Promise<void> | null = new Promise((resolve) => {
    release = resolve;
  });
  const { render } = hookHarness(async () => {
    const snapshot = sidebarRows(server);
    if (gate) await gate;
    return { rows: snapshot, nextCursor: null, hasMore: false, total: snapshot.length };
  });
  render();
  await tick();
  server = ["new", "old"];
  window.dispatchEvent(new Event("history"));
  await new Promise((resolve) => setTimeout(resolve, 350));
  gate = null;
  release();
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.deepEqual(
    render().rows.map((row) => row.id),
    ["new", "old"],
  );
});

test("a scroll after one failed page asks for that page again", async () => {
  let failNext = false;
  const { render } = hookHarness(async ({ limit, cursor }) => {
    if (failNext) {
      failNext = false;
      throw new Error("offline");
    }
    const offset = cursor ? Number(cursor) : 0;
    const end = offset + limit;
    return {
      rows: sidebarRows(Array.from({ length: limit }, (_, i) => `c${offset + i}`)),
      nextCursor: String(end),
      hasMore: end < 100,
      total: 100,
    };
  });
  render();
  await new Promise((resolve) => setTimeout(resolve, 10));
  failNext = true;
  await render().loadMore();
  await render().loadMore();
  assert.equal(render().rows.length, 40);
});

test("Select All's full read rejects when a page fails, so no partial list is taken for all", async () => {
  let calls = 0;
  const { render } = hookHarness(async ({ limit, cursor }) => {
    calls += 1;
    if (calls === 2) throw new Error("offline");
    const offset = cursor ? Number(cursor) : 0;
    const end = offset + limit;
    return {
      rows: sidebarRows(Array.from({ length: limit }, (_, i) => `c${offset + i}`)),
      nextCursor: String(end),
      hasMore: end < 100,
      total: 100,
    };
  });
  render();
  await new Promise((resolve) => setTimeout(resolve, 10));
  await assert.rejects(render().loadAll());
});
