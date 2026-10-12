// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";
import type * as ProgressiveRowsModule from "../src/components/progressive-rows.tsx";
import type * as RowPagesModule from "../src/features/chat/hooks/use-sidebar-row-pages.ts";
import {
  type RowPageResult,
  RowPager,
} from "../src/features/chat/utils/sidebar-row-pager.ts";
import * as rowQuery from "../src/features/chat/utils/sidebar-row-query.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const {
  applySidebarRowQuery,
  mergeManualOrderTail,
  sidebarRowQuery,
  sidebarRowQueryKey,
} = rowQuery;

type Row = { id: string; projectId?: string | null; updatedAt: number };

function rows(count: number, prefix = "c"): Row[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `${prefix}${i}`,
    updatedAt: count - i,
  }));
}

function fakeServer(all: () => Row[]) {
  const calls: { limit: number; cursor: string | null }[] = [];
  const fetchPage = async (
    _query: unknown,
    limit: number,
    cursor: string | null,
  ): Promise<RowPageResult<Row>> => {
    calls.push({ limit, cursor });
    const list = all();
    const offset = cursor ? Number(cursor) : 0;
    const end = offset + limit;
    return {
      rows: list.slice(offset, end),
      nextCursor: end < list.length ? String(end) : null,
      hasMore: end < list.length,
      total: list.length,
    };
  };
  return { calls, fetchPage };
}

function pager(
  fetchPage: ReturnType<typeof fakeServer>["fetchPage"],
  initialLimit = 20,
) {
  return new RowPager<unknown, Row>(fetchPage, {}, "q", {
    initialLimit,
    pageSize: 20,
  });
}

test("the first window asks for one page, and each further page for twenty", async () => {
  const server = fakeServer(() => rows(1000));
  const list = pager(server.fetchPage);
  await list.start();
  assert.equal(list.getState().rows.length, 20);
  assert.equal(list.getState().hasMore, true);
  assert.equal(list.getState().total, 1000);
  await list.loadMore();
  await list.loadMore();
  assert.deepEqual(server.calls, [
    { limit: 20, cursor: null },
    { limit: 20, cursor: "20" },
    { limit: 20, cursor: "40" },
  ]);
  assert.equal(list.getState().rows.length, 60);
  assert.equal(new Set(list.getState().rows.map((r) => r.id)).size, 60);
});

test("a list that fits in one page is exhausted and never asks again", async () => {
  for (const count of [0, 1, 19, 20]) {
    const server = fakeServer(() => rows(count));
    const list = pager(server.fetchPage);
    await list.start();
    assert.equal(list.getState().rows.length, count);
    assert.equal(list.getState().hasMore, false);
    await list.loadMore();
    assert.equal(server.calls.length, 1, `${count} rows`);
  }
});

test("twenty-one rows take exactly two pages; forty-one take three", async () => {
  for (const [count, pages] of [
    [21, 2],
    [40, 2],
    [41, 3],
  ] as const) {
    const server = fakeServer(() => rows(count));
    const list = pager(server.fetchPage);
    await list.start();
    while (list.getState().hasMore) await list.loadMore();
    assert.equal(list.getState().rows.length, count);
    assert.equal(server.calls.length, pages, `${count} rows`);
  }
});

test("a refresh re-reads the loaded window from the top in one request", async () => {
  let data = rows(500);
  const server = fakeServer(() => data);
  const list = pager(server.fetchPage);
  await list.start();
  await list.loadMore();
  await list.loadMore();
  server.calls.length = 0;
  data = [
    { id: "c300", updatedAt: 10_000 },
    ...data.filter((r) => r.id !== "c300"),
  ];
  await list.refresh();
  assert.deepEqual(server.calls, [{ limit: 60, cursor: null }]);
  assert.equal(list.getState().rows.length, 60);
  assert.equal(list.getState().rows[0].id, "c300");
});

test("a window past one request's cap is read in steps", async () => {
  const server = fakeServer(() => rows(1000));
  const list = new RowPager<unknown, Row>(server.fetchPage, {}, "q", {
    initialLimit: 20,
    pageSize: 20,
    maxRequestLimit: 200,
  });
  await list.loadAll();
  assert.equal(list.getState().rows.length, 1000);
  assert.equal(list.getState().hasMore, false);
  assert.ok(server.calls.every((call) => call.limit <= 200));
  server.calls.length = 0;
  await list.refresh();
  assert.deepEqual(
    server.calls.map((call) => call.limit),
    [200, 200, 200, 200, 200],
  );
});

test("a page landing after the query changed is dropped", async () => {
  let release: (() => void) | undefined;
  let slow = false;
  const answer = (prefix: string, limit: number): RowPageResult<Row> => ({
    rows: rows(limit, prefix),
    nextCursor: "next",
    hasMore: true,
  });
  const fetchPage = async (query: unknown, limit: number) => {
    const prefix = (query as { name: string }).name;
    if (slow) await new Promise<void>((resolve) => (release = resolve));
    return answer(prefix, limit);
  };
  const list = new RowPager<unknown, Row>(fetchPage, { name: "a" }, "a", {
    initialLimit: 20,
    pageSize: 20,
  });
  await list.start();
  slow = true;
  const late = list.loadMore();
  slow = false;
  await list.setQuery({ name: "b" }, "b");
  release?.();
  await late;
  assert.ok(list.getState().rows.every((row) => row.id.startsWith("b")));
  assert.equal(list.getState().rows.length, 20);
});

test("rows a page repeats are kept once", async () => {
  const fetchPage = async (
    _q: unknown,
    limit: number,
    cursor: string | null,
  ) => ({
    rows: cursor
      ? [{ id: "c19", updatedAt: 0 }, ...rows(limit, "d")]
      : rows(limit),
    nextCursor: "x",
    hasMore: true,
  });
  const list = new RowPager<unknown, Row>(fetchPage, {}, "q", {
    initialLimit: 20,
    pageSize: 20,
  });
  await list.start();
  await list.loadMore();
  const ids = list.getState().rows.map((row) => row.id);
  assert.equal(ids.length, new Set(ids).size);
  assert.equal(ids.length, 40);
});

test("a cursor the server rejects re-reads the same depth from the top", async () => {
  const server = fakeServer(() => rows(100));
  let reject = false;
  const fetchPage: typeof server.fetchPage = async (q, limit, cursor) => {
    if (reject && cursor) {
      reject = false;
      throw Object.assign(new Error("stale"), { cursorRejected: true });
    }
    return server.fetchPage(q, limit, cursor);
  };
  const list = pager(fetchPage);
  await list.start();
  await list.loadMore();
  reject = true;
  server.calls.length = 0;
  await list.loadMore();
  assert.equal(list.getState().error, null);
  assert.equal(list.getState().rows.length, 40);
  assert.deepEqual(server.calls, [{ limit: 40, cursor: null }]);
});

test("a failed page stops paging until retried, and keeps the rows it had", async () => {
  let fail = false;
  const server = fakeServer(() => rows(100));
  const fetchPage: typeof server.fetchPage = async (q, limit, cursor) => {
    if (fail) throw new Error("offline");
    return server.fetchPage(q, limit, cursor);
  };
  const list = pager(fetchPage);
  await list.start();
  fail = true;
  await list.loadMore();
  assert.equal(list.getState().rows.length, 20);
  assert.ok(list.getState().error);
  fail = false;
  const before = server.calls.length;
  await list.loadMore();
  assert.equal(server.calls.length, before, "an errored list kept asking");
  await list.retry();
  assert.equal(list.getState().rows.length, 40);
  assert.equal(list.getState().error, null);
});

test("the full-list fallback filters and orders like the server", () => {
  const all: Row[] = [
    { id: "a", updatedAt: 9, projectId: null },
    { id: "b", updatedAt: 8, projectId: "p" },
    { id: "c", updatedAt: 7, projectId: null },
    { id: "d", updatedAt: 6, projectId: "p" },
    { id: "e", updatedAt: 5, projectId: null },
  ];
  const ids = (query: Partial<rowQuery.SidebarRowQuery>) =>
    applySidebarRowQuery(all, sidebarRowQuery(query)).map((row) => row.id);
  assert.deepEqual(ids({}), ["a", "b", "c", "d", "e"]);
  assert.deepEqual(ids({ unfiledOnly: true, excludeIds: ["c"] }), ["a", "e"]);
  assert.deepEqual(ids({ projectId: "p" }), ["b", "d"]);
  assert.deepEqual(ids({ includeIds: ["e", "a"] }), ["a", "e"]);
  assert.deepEqual(ids({ sort: "manual", manualOrder: ["d", "a"] }), [
    "b",
    "c",
    "e",
    "d",
    "a",
  ]);
  assert.deepEqual(ids({ sort: "updated", manualOrder: ["e"] }), [
    "a",
    "b",
    "c",
    "d",
    "e",
  ]);
});

test("the same lists are the same query however the stores ordered them", () => {
  assert.equal(
    sidebarRowQueryKey(sidebarRowQuery({ excludeIds: ["b", "a", "b"] })),
    sidebarRowQueryKey(sidebarRowQuery({ excludeIds: ["a", "b"] })),
  );
});

test("a manual order written from loaded rows keeps every unloaded row after them", () => {
  assert.deepEqual(
    mergeManualOrderTail(["c", "a", "b"], ["a", "b", "c", "d", "e"]),
    ["c", "a", "b", "d", "e"],
  );
  assert.deepEqual(mergeManualOrderTail(["x"], []), ["x"]);
});

function loadRowPages(options: {
  server: "paged" | "old";
  legacyDone?: boolean;
}) {
  const requests: unknown[] = [];
  let fullReads = 0;
  const threads = Array.from({ length: 45 }, (_, i) => ({
    id: `t${i}`,
    title: `chat ${i}`,
    modelType: "base",
    archived: i % 9 === 8,
    createdAt: 1,
    updatedAt: 1000 - i,
    projectId: i % 5 === 4 ? "p" : null,
  }));
  const groupThreads = (list: typeof threads, archived = false) =>
    list
      .filter((t) => Boolean(t.archived) === archived)
      .map((t) => ({
        type: "single" as const,
        id: t.id,
        threadIds: [t.id],
        title: t.title,
        createdAt: t.createdAt,
        updatedAt: t.updatedAt,
        projectId: t.projectId,
      }));
  const mod = loadWithStubs<typeof RowPagesModule>(
    new URL(
      "../src/features/chat/hooks/use-sidebar-row-pages.ts",
      import.meta.url,
    ),
    {
      react: React,
      "@/features/auth": { getAuthSessionEpoch: () => 0 },
      "../api/chat-api": {
        CHAT_HISTORY_UPDATED_EVENT: "x",
        fetchSidebarRows: async (request: {
          limit: number;
          cursor: string | null;
        }) => {
          requests.push(request);
          if (options.server === "old") return null;
          const offset = request.cursor ? Number(request.cursor) : 0;
          const end = offset + request.limit;
          return {
            rows: [
              {
                id: `s${offset}`,
                type: "single",
                threadIds: ["gone"],
                title: "",
                createdAt: 1,
                updatedAt: 1,
              },
            ],
            nextCursor: String(end),
            hasMore: true,
            total: 99,
          };
        },
      },
      "../utils/chat-history-storage": {
        listStoredChatThreads: async () => {
          fullReads += 1;
          return threads;
        },
        settleLegacyChatImport: async () => options.legacyDone ?? true,
      },
      "../utils/chat-thread-tombstones": {
        isChatThreadDeleted: (id: string) => id === "gone",
      },
      "../utils/sidebar-row-pager": { RowPager },
      "../utils/sidebar-row-query": rowQuery,
      "./use-chat-sidebar-items": { groupThreads },
    },
  );
  return { mod, requests, fullReads: () => fullReads };
}

test("an old server is answered from the full list, sliced in pages, read once", async () => {
  const { mod, requests, fullReads } = loadRowPages({ server: "old" });
  const query = sidebarRowQuery({ unfiledOnly: true });
  const first = await mod.fetchSidebarRowPage(query, 20, null);
  assert.equal(requests.length, 1, "the paged route was not tried first");
  assert.equal(first.rows.length, 20);
  assert.ok(first.rows.every((row) => !row.projectId));
  assert.equal(first.hasMore, true);
  const second = await mod.fetchSidebarRowPage(query, 20, first.nextCursor);
  assert.equal(requests.length, 1);
  assert.equal(
    new Set([...first.rows, ...second.rows].map((r) => r.id)).size,
    first.rows.length + second.rows.length,
  );
  assert.equal(first.total, 32);
  assert.equal(second.hasMore, false);
  assert.equal(fullReads(), 1, "each page re-read the full list");
});

test("until the browser-history import is done, the sidebar lists through the full list", async () => {
  const { mod, requests } = loadRowPages({
    server: "paged",
    legacyDone: false,
  });
  const page = await mod.fetchSidebarRowPage(sidebarRowQuery({}), 20, null);
  assert.equal(requests.length, 0);
  assert.equal(page.rows.length, 20);
});

test("a paged server's rows pass through, minus chats already deleted here", async () => {
  const { mod, requests } = loadRowPages({ server: "paged" });
  const page = await mod.fetchSidebarRowPage(sidebarRowQuery({}), 20, null);
  assert.equal(requests.length, 1);
  assert.deepEqual(page.rows, []);
  assert.equal(page.hasMore, true);
  assert.equal(page.nextCursor, "20");
  assert.equal(page.total, 99);
});

const { ProgressiveRows } = loadWithStubs<typeof ProgressiveRowsModule>(
  new URL("../src/components/progressive-rows.tsx", import.meta.url),
  { react: React, "react/jsx-runtime": jsxRuntime },
);

function renderRows(count: number, hasMore: boolean): string {
  const items = Array.from({ length: count }, (_, i) => ({ id: `chat-${i}` }));
  return renderToStaticMarkup(
    React.createElement(
      "ul",
      null,
      React.createElement(ProgressiveRows<{ id: string }>, {
        items,
        pageSize: 20,
        hasMore,
        onReachEnd: () => undefined,
        renderItem: ({ id }) => React.createElement("li", { "data-row": id }),
        end: React.createElement("li", { "data-end": "" }),
      }),
    ),
  );
}

test("a loaded page with more on the server ends in the sentinel, not the end strip", () => {
  const html = renderRows(20, true);
  assert.equal(html.match(/data-row=/g)?.length, 20);
  assert.doesNotMatch(html, /data-end/);
  assert.match(html.slice(html.lastIndexOf("<li")), /^<li aria-hidden="true"/);
  assert.match(renderRows(20, false), /<li data-end=""><\/li><\/ul>$/);
});

test("a server page is asked for only after a scroll, one page per scroll", () => {
  const src = readSrc("components/progressive-rows.tsx");
  const advance = src.slice(
    src.indexOf("const advance = useCallback("),
    src.indexOf("// re-observe"),
  );
  assert.match(
    advance,
    /if \(!fetchMore \|\| fetchingRef\.current \|\| !root \|\| !armedRef\.current\) return;\n\s*armedRef\.current = false;/,
  );
  assert.ok(
    advance.indexOf("items.length > limit") < advance.indexOf("armedRef"),
    "local mounting waits for a scroll",
  );
  // the arm is per list: one list's scroll-armed state cannot make another list fetch
  assert.doesNotMatch(src, /new WeakSet/);
  assert.match(src, /const onArm = \(\) => \{\n\s*armedRef\.current = true;/);
  assert.match(
    src,
    /const ARMING_EVENTS = \["scroll", "wheel", "touchmove"\] as const;/,
  );
});

test("the sidebar reads every chat list through the pager, twenty rows at a time", () => {
  const src = readSrc("components/app-sidebar.tsx");
  assert.doesNotMatch(src, /useChatSidebarItems\(/);
  assert.match(src, /const RECENTS_PAGE_SIZE = SIDEBAR_PAGE_SIZE;/);
  assert.match(src, /= useSidebarRowPages\(\n\s*chatRowSlots,/);
  for (const list of [
    "pinnedRows",
    "rows",
    "visibleProjectRecords",
    "visibleChats",
    "runItems",
    "sortedRecentChatItems",
  ]) {
    assert.match(
      src,
      new RegExp(`<ProgressiveRows\\n\\s*items=\\{${list}\\}`),
      list,
    );
  }
  assert.equal(
    readSrc("features/chat/utils/sidebar-row-query.ts").includes(
      "SIDEBAR_PAGE_SIZE = 20",
    ),
    true,
  );
});

test("the thread-row lookup reads the key and the per-request cap the route defines", () => {
  const route = readFileSync(
    new URL("../../backend/routes/chat_history.py", import.meta.url),
    "utf8",
  );
  const client = readSrc("features/chat/api/chat-api.ts");
  assert.match(route, /rowIdByThreadId: dict\[str, str\]/);
  assert.match(client, /data\.rowIdByThreadId/);
  const serverCap = route.match(/_SIDEBAR_THREAD_ROWS_MAX = (\d+)/)?.[1];
  const clientCap = client.match(/const SIDEBAR_THREAD_ROWS_MAX = (\d+)/)?.[1];
  assert.ok(serverCap && clientCap, "both caps are declared");
  assert.ok(Number(clientCap) <= Number(serverCap), "client chunks within the route's cap");
});
