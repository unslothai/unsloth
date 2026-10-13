// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as RowPagesModule from "../src/features/chat/hooks/use-sidebar-row-pages.ts";
import {
  keepOpenedChats,
  OPENED_CHATS_LIMIT,
  touchOpenedChat,
} from "../src/features/chat/utils/opened-chats.ts";
import { RowPager } from "../src/features/chat/utils/sidebar-row-pager.ts";
import * as rowQuery from "../src/features/chat/utils/sidebar-row-query.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const none = new Set<string>();

function open(ids: string[], running: ReadonlySet<string> = none, start: string[] = []) {
  let opened = start;
  for (const id of ids) opened = keepOpenedChats(touchOpenedChat(opened, id), running);
  return opened;
}

const ids = (n: number, prefix = "c") => Array.from({ length: n }, (_, i) => `${prefix}${i}`);

test("the cap is 20 and the most recently opened chat comes first", () => {
  assert.equal(OPENED_CHATS_LIMIT, 20);
  assert.deepEqual(open(["a", "b", "c"]), ["c", "b", "a"]);
});

test("opening a 21st chat evicts the least recently opened one", () => {
  const opened = open(ids(21));
  assert.equal(opened.length, 20);
  assert.equal(opened[0], "c20");
  assert.ok(!opened.includes("c0"), "the oldest opened chat is evicted");
  assert.ok(opened.includes("c1"));
});

test("reopening a chat moves it to the front instead of evicting it", () => {
  const opened = open(["c0"], none, open(ids(20)));
  assert.equal(opened.length, 20);
  assert.equal(opened[0], "c0");
  // the next new chat evicts c1, now the least recently opened, not c0
  const next = open(["new"], none, opened);
  assert.ok(next.includes("c0"));
  assert.ok(!next.includes("c1"));
});

test("a running chat is never evicted; the next oldest idle one goes instead", () => {
  const running = new Set(["c0"]);
  const opened = open(ids(25), running);
  assert.equal(opened.length, 20);
  assert.ok(opened.includes("c0"), "the running chat stays");
  for (const gone of ["c1", "c2", "c3", "c4", "c5"]) assert.ok(!opened.includes(gone), gone);
});

test("with more than 20 running chats the list passes the cap rather than evict one", () => {
  const all = ids(23);
  const running = new Set(all);
  const opened = open(all, running);
  assert.equal(opened.length, 23);
  // once one stops, the next open brings the list back toward the cap
  const after = open(["x"], new Set(all.slice(1)), opened);
  assert.ok(!after.includes("c0"), "the stopped chat is the first evicted");
  assert.equal(after[0], "x", "the chat just opened is never the one evicted");
  assert.equal(after.length, 23);
});

test("eviction is deterministic: same opens, same running set, same list", () => {
  const sequence = [...ids(30), "c3", "c29", "c10", ...ids(5, "d")];
  const running = new Set(["c2", "c7"]);
  assert.deepEqual(open(sequence, running), open(sequence, running));
});

/** The hook under a minimal React: state cells by call order, effects re-run on changed deps. */
function renderHook(slots: Record<string, unknown>) {
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
      const same = c.v?.deps && deps && deps.every((dep, k) => Object.is(dep, c.v?.deps?.[k]));
      if (!same) {
        c.v = { deps };
        fn();
      }
    },
  };
  const rows = (prefix: string, n: number) =>
    Array.from({ length: n }, (_, i) => ({
      id: `${prefix}${i}`,
      type: "single",
      threadIds: [`${prefix}${i}`],
      title: "",
      createdAt: 0,
      updatedAt: n - i,
    }));
  const mod = loadWithStubs<typeof RowPagesModule>(
    new URL("../src/features/chat/hooks/use-sidebar-row-pages.ts", import.meta.url),
    {
      react,
      "@/features/auth": { getAuthSessionEpoch: () => 0 },
      "../api/chat-api": {
        CHAT_HISTORY_UPDATED_EVENT: "history",
        fetchSidebarRows: async (request: { includeIds: string[] | null; limit: number }) => {
          const list = request.includeIds
            ? rows("old", 1).map((row) => ({ ...row, id: request.includeIds?.[0] ?? row.id }))
            : rows("r", request.limit);
          return { rows: list, nextCursor: null, hasMore: false, total: list.length };
        },
      },
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
  return () => {
    at = 0;
    return mod.useSidebarRowPages(slots as never, true);
  };
}

test("an opened chat past Recents' loaded page is looked up but never listed", async () => {
  const render = renderHook({
    recents: { query: rowQuery.sidebarRowQuery({}) },
    active: {
      query: rowQuery.sidebarRowQuery({ includeIds: ["deep-chat"] }),
      initialLimit: Number.MAX_SAFE_INTEGER,
      lookupOnly: true,
    },
  });
  render();
  await new Promise((resolve) => setTimeout(resolve, 20));
  const { items, listedItems } = render();
  assert.ok(items.some((row) => row.id === "deep-chat"), "lookups see the opened chat");
  assert.ok(!listedItems.some((row) => row.id === "deep-chat"), "no list draws it out of order");
  assert.deepEqual(
    listedItems.map((row) => row.id),
    Array.from({ length: 20 }, (_, i) => `r${i}`),
  );
});
