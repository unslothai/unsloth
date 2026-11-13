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

test("a refresh that reads the same rows renders nothing; one that reads a change renders once", async () => {
  let data = rows(100);
  const server = fakeServer(() => data.map((row) => ({ ...row })));
  const list = pager(server.fetchPage);
  await list.start();
  const before = list.getState().rows;
  let renders = 0;
  list.subscribe(() => {
    renders += 1;
  });
  await list.refresh();
  assert.equal(renders, 0);
  assert.equal(list.getState().rows, before, "an unchanged window keeps its identity");
  data = [{ id: "new", updatedAt: 1_000 }, ...data];
  await list.refresh();
  assert.equal(renders, 1);
  assert.equal(list.getState().rows[0].id, "new");
});

test("a page asked for while a history refresh is in flight loads after it, not never", async () => {
  const server = fakeServer(() => rows(200));
  let gate: Promise<void> | null = null;
  let release = () => {};
  const list = pager(async (query, limit, cursor) => {
    if (gate) await gate;
    return server.fetchPage(query, limit, cursor);
  });
  await list.start();
  gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const refresh = list.refresh();
  const more = list.loadMore();
  gate = null;
  release();
  await Promise.all([refresh, more]);
  assert.equal(list.getState().rows.length, 40);
  // two scrolls during one page still ask for one page each, never more
  await Promise.all([list.loadMore(), list.loadMore()]);
  assert.equal(list.getState().rows.length, 60);
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
    if (slow) {
      await new Promise<void>((resolve) => {
        release = resolve;
      });
    }
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

test("a cursor the server rejects re-reads from the top through the page it asked for", async () => {
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
  assert.equal(list.getState().rows.length, 60);
  assert.deepEqual(server.calls, [{ limit: 60, cursor: null }]);
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
  /** Answers the paged route instead of the default endless single-row pages. */
  pages?: (cursor: string | null) => unknown;
}) {
  const requests: unknown[] = [];
  const repaired: { id: string; title: string }[][] = [];
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
          if (options.pages) return options.pages(request.cursor);
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
      "../utils/repair-legacy-chat-titles": {
        repairLegacyChatTitles: async (list: { id: string; title: string }[]) => {
          repaired.push(list);
          return 0;
        },
      },
      "../utils/chat-thread-tombstones": {
        isChatThreadDeleted: (id: string) => id === "gone",
      },
      "../utils/sidebar-row-pager": { RowPager },
      "../utils/sidebar-row-query": rowQuery,
      "./use-chat-sidebar-items": { groupThreads },
    },
  );
  return { mod, requests, repaired, fullReads: () => fullReads };
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

test("an old server's folders take their activity from the full list the lists already read", async () => {
  const { mod, fullReads } = loadRowPages({ server: "old" });
  await mod.fetchSidebarRowPage(sidebarRowQuery({}), 20, null);
  // t4 is the newest unarchived chat in "p"; t44 is archived and does not count
  assert.deepEqual([...(await mod.readFullListProjectActivity())], [["p", 996]]);
  assert.equal(fullReads(), 1, "the folder activity read the history again");
  const projects = readSrc("features/chat/hooks/use-chat-projects.ts");
  assert.match(projects, /if \(projects\.every\(\(project\) => "lastActivityAt" in project\)\) return projects;/);
  assert.match(projects, /nextProjects = Array\.isArray\(next\) \? await withChatActivity\(next\) : \[\];/);
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

test("legacy titles are repaired from the rows a page loads, and from the fallback's full list", async () => {
  const paged = loadRowPages({ server: "paged" });
  await paged.mod.fetchSidebarRowPage(sidebarRowQuery({}), 20, null);
  assert.deepEqual(paged.repaired, [[{ id: "gone", title: "" }]]);
  const old = loadRowPages({ server: "old" });
  await old.mod.fetchSidebarRowPage(sidebarRowQuery({}), 20, null);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(old.repaired.length, 1);
  assert.equal(old.repaired[0].length, 45);
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
  assert.doesNotMatch(src, /new WeakSet/);
  // a scroll while a page is in flight must not arm the next one
  assert.match(
    src,
    /\n(?:\s*\/\/.*\n)?\s*if \(fetchingRef\.current \|\| performance\.now\(\) - drawnAtRef\.current < GESTURE_TAIL_MS\) return;\n\s*armedRef\.current = true;/,
  );
  assert.match(src, /const down = top > lastTop;\n\s*lastTop = top;\n\s*if \(!down\) return;/);
  assert.match(src, /event\.type === "wheel" && \(event as WheelEvent\)\.deltaY <= 0\) \{\n\s*return;/);
  // a touch arms only when the finger moves up (the list scrolls down), not on a swipe up or sideways
  assert.match(src, /const down = touchY !== null && y !== undefined && y < touchY;\n\s*if \(y !== undefined\) touchY = y;\n\s*if \(!down\) return;/);
  assert.match(src, /touchY = \(event as TouchEvent\)\.touches\[0\]\?\.clientY \?\? null;\n\s*return;/);
  assert.match(src, /fetchingRef\.current = false;\n\s*drawnAtRef\.current = performance\.now\(\);/);
  assert.match(
    src,
    /const ARMING_EVENTS = \["scroll", "wheel", "touchstart", "touchmove"\] as const;/,
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

test("a drag in a partly loaded list shows its order at once and saves it once every id is known", () => {
  const src = readSrc("components/app-sidebar.tsx");
  const body = src.slice(
    src.indexOf("const writeManualOrder = useCallback("),
    src.indexOf("// Escape leaves a selection"),
  );
  const partial = body.slice(body.indexOf("const saved ="));
  assert.ok(
    partial.indexOf("setPendingManualOrder(") < partial.indexOf("readAllSidebarRowIds("),
  );
  assert.equal(partial.slice(0, partial.indexOf("readAllSidebarRowIds(")).match(/setManualOrder\(scope/g)?.length ?? 0, 1);
  assert.doesNotMatch(partial, /settle\(shown\)/);
  assert.match(partial, /if \(attempt >= 2\) throw error;/);
  assert.match(partial, /setPendingManualOrder\(\(\{ \[scope\]: _dropped, \.\.\.rest \}\) => rest\);\n\s*toast\.error\("Couldn't save the new order"\);/);
  assert.match(src, /applyManualOrder\(items, displayManualOrder\[scope\]/);
});

test("the full-list fallback finds a compare row through one of its pane threads", () => {
  const rows = [
    { id: "pair", threadIds: ["pane-a", "pane-b"], updatedAt: 2 },
    { id: "solo", threadIds: ["solo"], updatedAt: 1 },
  ];
  const found = applySidebarRowQuery(rows, sidebarRowQuery({ includeIds: ["pane-b"] }));
  assert.deepEqual(found.map((row) => row.id), ["pair"]);
});

test("the last 20 opened chats and every running one load their rows, so row lookups never miss them", () => {
  const src = readSrc("components/app-sidebar.tsx");
  assert.match(
    src,
    /\[slotActiveId, \.\.\.openedChatIds, \.\.\.runningThreadKey\.split\("\\n"\)\]\.filter\(/,
  );
  assert.match(src, /isAssistantLocalThreadId\(id\),\n\s*\),\n\s*\),\n\s*\]\.sort\(\);/);
  assert.match(
    src,
    /keepOpenedChats\(touchOpenedChat\(opened, slotActiveId\), running\)/,
  );
  // looked up, never drawn: an opened chat past a list's loaded pages stays out of that list
  assert.match(src, /includeIds: lookupIds \}\),\n\s*initialLimit: Number\.MAX_SAFE_INTEGER,\n\s*lookupOnly: true,/);
  assert.match(src, /Boolean\(id\) && !isAssistantLocalThreadId\(id\)/);
  for (const list of ["recentChatItems", "pinnedChatItems"]) {
    const body = src.slice(src.indexOf(`const ${list} = useMemo(`), src.indexOf(`const ${list} = useMemo(`) + 400);
    assert.match(body, /listedChatItems/, list);
    assert.doesNotMatch(body, /allChatItems/, list);
  }
  assert.match(src, /for \(const \[projectId, items\] of listedChatsByProjectId\)/);
  assert.match(src, /for \(const item of listedChatItems\) \{\n\s*const sectionId = sectionByChatId\[item\.id\];/);
});

test("follow-ups that main got from the full list still happen with paged lists", () => {
  const src = readSrc("components/app-sidebar.tsx");
  // a chord at the last loaded row of any paged list reads its next page; a click there does not
  assert.match(src, /chordTargetRef\.current = item\?\.id \?\? null;\n\s*openChatItemById\(item\);/);
  assert.equal(src.match(/if \(!activeThreadId \|\| chordTargetRef\.current !== activeThreadId\) return;/g)?.length, 2);
  assert.match(src, /if \(list\.rows\[list\.rows\.length - 1\]\?\.id === activeThreadId\) void list\.loadMore\(\);/);
  assert.match(src, /window\.addEventListener\(CHAT_HISTORY_UPDATED_EVENT, probe\);/);
  const write = src.slice(src.indexOf("const writeManualOrder = useCallback("));
  assert.ok(write.indexOf("writes.set(scope, generation);") < write.indexOf("const slot ="));
  // project activity follows chat activity, not only project edits
  const projects = readSrc("features/chat/hooks/use-chat-projects.ts");
  assert.match(projects, /const onActivity = \(\) => onChatActivity\(\);/);
  assert.match(projects, /window\.addEventListener\(CHAT_HISTORY_UPDATED_EVENT, onActivity\);/);
  assert.match(projects, /window\.removeEventListener\(CHAT_HISTORY_UPDATED_EVENT, onActivity\);/);
  assert.match(projects, /if \(projectsLoaded && JSON\.stringify\(projects\) === JSON\.stringify\(cachedProjects\)\) return;/);
});

test("\"No chats yet\" asks for any chat, archived or in an unloaded folder, before it shows", () => {
  const src = readSrc("components/app-sidebar.tsx");
  assert.match(src, /\[sidebarRowQuery\(\{\}\), sidebarRowQuery\(\{ archived: true \}\)\]/);
  // and shows nothing until a probe has answered, so it never flashes while one is in flight
  assert.match(src, /useState<boolean \| null>\(null\);/);
  assert.match(src, /hasUnlistedChats === false && \(/);
  assert.match(src, /\.then\(\(\) => setSelectAllPending\(true\)\)\n\s*\.catch\(\(\) => undefined\);/);
});

test("a hidden or hand-ordered section still loads its newest chat, which orders Move to", () => {
  const src = readSrc("components/app-sidebar.tsx");
  assert.match(src, /\(hiddenSectionSet\.has\(section\.id\) \|\|\n\s*section\.sort === "manual" \|\|\n\s*ids\.length < members\.length\)/);
  assert.match(src, /slots\[`latest:\$\{section\.id\}`\] = \{\n\s*initialLimit: 1,\n\s*query: sidebarRowQuery\(\{ includeIds: members \}\),\n\s*lookupOnly: true,/);
  const latest = src.indexOf("slots[`latest:${section.id}`]");
  assert.ok(latest > 0 && latest < src.indexOf("if (ids.length === 0 || hiddenSectionSet.has(section.id)) continue;"));
  // it is never paged past that row
  assert.match(src, /name === "recents" \|\| name\.startsWith\("latest:"\) \|\| !list\.hasMore/);
});

test("a failed first read never shows \"No chats yet\"; it waits for a read that worked, as main did", () => {
  const src = readSrc("components/app-sidebar.tsx");
  assert.match(src, /const chatItemsLoaded = Boolean\(recentsPages\?\.loaded && !recentsPages\.error\);/);
});

test("a section's Mark all read and Archive act on the whole section, not its loaded pages", () => {
  const src = readSrc("components/app-sidebar.tsx");
  const menu = src.slice(src.indexOf("const unreadInSection"), src.indexOf("const unreadInSection") + 1600);
  assert.match(menu, /inSection\(unreadRowIds\[threadId\] \?\? threadId\)/);
  assert.match(menu, /onSelect=\{\(\) => clearThreadsUnread\(unreadInSection\)\}/);
  assert.match(menu, /sectionPages\?\.hasMore && sectionQuery \? await readAllSidebarRows\(sectionQuery\) : chats/);
});

test("only folder rows on screen read their chats", () => {
  const src = readSrc("components/app-sidebar.tsx");
  assert.match(src, /!collapsedProjectIds\.has\(project\.id\) && mountedFolderIds\.has\(project\.id\)/);
  assert.match(src, /<MountedFolder id=\{project\.id\} onChange=\{setFolderMounted\} \/>/);
});

test("a list that shrank while a page was read frees its paging latch", () => {
  const src = readSrc("components/progressive-rows.tsx");
  assert.match(src, /if \(itemCountRef\.current <= fetchFromRef\.current\) fetchingRef\.current = false;/);
});

test("a running Compare pane keeps its opened pair row, which is opened by pair id", () => {
  const src = readSrc("components/app-sidebar.tsx");
  assert.match(src, /if \(item\.threadIds\?\.some\(\(id\) => running\.has\(id\)\)\) rows\.add\(item\.id\);/);
  assert.match(src, /const running = new Set\(runningRowKey\.split\("\\n"\)\);\n\s*setOpenedChatIds\(\(opened\) => keepOpenedChats\(touchOpenedChat\(opened, slotActiveId\), running\)\);/);
});

test("a page emptied by chats deleted here does not end the list: the next page still loads", async () => {
  const fetchPage = async (_q: unknown, limit: number, cursor: string | null) => {
    const at = cursor ? Number(cursor) : 0;
    return {
      rows: at === 0 ? [] : rows(limit, `p${at}-`),
      nextCursor: String(at + limit),
      hasMore: at < 60,
    };
  };
  const list = new RowPager<unknown, Row>(fetchPage, {}, "q", { initialLimit: 20, pageSize: 20 });
  await list.start();
  assert.equal(list.getState().rows.length, 20, "the emptied first page ended the read");
  assert.equal(list.getState().rows[0].id, "p20-0");
  const stuck = new RowPager<unknown, Row>(
    async () => ({ rows: [], nextCursor: "same", hasMore: true }),
    {},
    "q",
    { initialLimit: 20, pageSize: 20 },
  );
  await stuck.start();
  assert.equal(stuck.getState().rows.length, 0);
});

test("a manual list still paging holds back folders and pages ranked past its last loaded chat", () => {
  const { holdRowsPastLoadedChats } = rowQuery;
  const order = ["c1", "f1", "c2", "c3", "f2", "c4", "p1"];
  const shown = [
    { id: "f0", kind: "project" }, // unranked folders lead
    { id: "c1", kind: "chat" },
    { id: "f1", kind: "project" },
    { id: "c2", kind: "chat" },
    { id: "f2", kind: "project" },
    { id: "p1", kind: "page" },
  ];
  // c3 and c4 are not loaded yet: f2 and p1 come after c3 on main, so they wait
  assert.deepEqual(
    holdRowsPastLoadedChats(shown, order, true).map((row) => row.id),
    ["f0", "c1", "f1", "c2"],
  );
  assert.deepEqual(holdRowsPastLoadedChats(shown, order, false).map((row) => row.id), shown.map((row) => row.id));
  assert.deepEqual(
    holdRowsPastLoadedChats([{ id: "f0", kind: "project" }, { id: "u1", kind: "chat" }, { id: "p0", kind: "page" }], order, true).map((row) => row.id),
    ["f0", "u1"],
  );
});

test("a drag saved from a partly loaded manual list keeps held-back rows where they were", () => {
  const { settledManualOrder } = rowQuery;
  const current = ["c1", "f1", "c2", "c3", "f2", "c4"];
  const all = ["u1", "c1", "c2", "c3", "c4"]; // the server's chat order: unranked u1 first
  assert.deepEqual(settledManualOrder(["u1", "c2", "c1", "f1"], all, current), [
    "u1",
    "c2",
    "c1",
    "f1",
    "c3",
    "f2",
    "c4",
  ]);
  // an unloaded unranked chat (shown first on main) goes ahead of the saved tail, not after it
  assert.deepEqual(settledManualOrder(["c2", "c1"], ["u1", "u2", "c1", "c2", "c3"], ["c1", "c2", "c3"]), [
    "c2",
    "c1",
    "u1",
    "u2",
    "c3",
  ]);
});

test("a training refresh past one request re-reads from the top if the run count moved under it", () => {
  const src = readSrc("features/training/hooks/use-training-history-sidebar.ts");
  assert.match(src, /for \(let attempt = 0; attempt < 3; attempt\+\+\) \{/);
  assert.match(src, /shifted \|\|= result\.total !== firstTotal \|\| fresh\.length < result\.runs\.length;/);
  assert.match(src, /runs\.push\(\.\.\.fresh\);/);
  assert.match(src, /if \(!shifted\) break;/);
});

test("Pinned and manual sections hold rows past their last loaded chat, and a drag saves around them", () => {
  const src = readSrc("components/app-sidebar.tsx");
  assert.match(src, /holdRowsPastLoadedChats\(\n\s*applyManualOrder\(rows, order, \(row\) => row\.id\),\n\s*order,\n\s*Boolean\(pinnedHasMore\),/);
  assert.match(src, /holdRowsPastLoadedChats\(\n\s*applyManualOrder\(list, order, \(row\) => row\.id\),\n\s*order,\n\s*sectionHasMore\.has\(section\.id\),/);
  assert.match(src, /settle\(settledManualOrder\(ids, all, current\)\);/);
});

test("a page asked for just before a history refresh is read by that refresh instead of dropped", async () => {
  let release: (() => void) | null = null;
  const fetchPage = async (_q: unknown, limit: number, cursor: string | null) => {
    const at = cursor ? Number(cursor) : 0;
    if (cursor && !release) await new Promise<void>((resolve) => { release = resolve; });
    return { rows: rows(100).slice(at, at + limit), nextCursor: String(at + limit), hasMore: at + limit < 100 };
  };
  const list = new RowPager<unknown, Row>(fetchPage, {}, "q", { initialLimit: 20, pageSize: 20 });
  await list.start();
  const page = list.loadMore();
  await new Promise((resolve) => setTimeout(resolve, 0));
  const refreshed = list.refresh();
  (release as (() => void) | null)?.();
  await Promise.all([page, refreshed]);
  assert.equal(list.getState().rows.length, 40, "the refresh re-read only the 20 already drawn");
});

test("a Compare row's panes are repaired with their own titles, not the row's", async () => {
  const src = readSrc("features/chat/hooks/use-sidebar-row-pages.ts");
  assert.match(src, /title: row\.threadTitles\?\.\[index\] \?\? row\.title,/);
});

test("a training refresh moves its offset by the server rows read, and a repeated run restarts it", () => {
  const src = readSrc("features/training/hooks/use-training-history-sidebar.ts");
  assert.match(src, /offset \+= result\.runs\.length;/);
  assert.match(src, /shifted \|\|= result\.total !== firstTotal \|\| fresh\.length < result\.runs\.length;/);
  assert.match(src, /if \(result\.runs\.length === 0 \|\| offset >= result\.total\) break;/);
});

test("a failed empty-history probe stops an older answer from showing \"No chats yet\"", () => {
  const src = readSrc("components/app-sidebar.tsx");
  assert.match(src, /\.catch\(\(\) => \{\n\s*if \(!cancelled && mine === latest\) setHasUnlistedChats\(null\);\n\s*\}\);/);
  assert.match(src, /const mine = \+\+latest;/);
  assert.match(src, /if \(!cancelled && mine === latest\) \{\n\s*setHasUnlistedChats\(pages\.some/);
});

test("Pinned and custom sections mount every row they have, so navigation sees only drawn rows", () => {
  const items = Array.from({ length: 27 }, (_, i) => ({ id: `row-${i}` }));
  const html = (mountAll: boolean) =>
    renderToStaticMarkup(
      React.createElement(
        "ul",
        null,
        React.createElement(ProgressiveRows<{ id: string }>, {
          items,
          pageSize: 20,
          mountAll,
          hasMore: true,
          onReachEnd: () => undefined,
          renderItem: ({ id }) => React.createElement("li", { "data-row": id }),
        }),
      ),
    );
  assert.equal(html(false).match(/data-row=/g)?.length, 20);
  assert.equal(html(true).match(/data-row=/g)?.length, 27);
  const src = readSrc("components/app-sidebar.tsx");
  assert.match(src, /items=\{pinnedRows\}\n\s*pageSize=\{SIDEBAR_PAGE_SIZE\}\n\s*mountAll\n/);
  assert.match(src, /items=\{rows\}\n\s*pageSize=\{SIDEBAR_PAGE_SIZE\}\n\s*mountAll\n/);
});

test("an empty spell starts with no answer from an earlier one, from its first paint", () => {
  const src = readSrc("components/app-sidebar.tsx");
  assert.match(src, /if \(!sidebarLooksEmpty\) \{\n\s*setHasUnlistedChats\(null\);\n\s*return;\n\s*\}/);
});

test("a whole-list read goes on past a page emptied by chats deleted here", async () => {
  const row = (id: string) => ({ id, type: "single", threadIds: [id], title: "", createdAt: 1, updatedAt: 1 });
  const { mod } = loadRowPages({
    server: "paged",
    pages: (cursor) =>
      cursor === null
        ? { rows: [row("gone")], nextCursor: "a", hasMore: true }
        : cursor === "a"
          ? { rows: [row("k1"), row("k2")], nextCursor: "b", hasMore: true }
          : { rows: [row("k3")], nextCursor: null, hasMore: false },
  });
  assert.deepEqual(await mod.readAllSidebarRowIds(sidebarRowQuery({})), ["k1", "k2", "k3"]);
  // an empty page that does not move the cursor still ends the read
  const stuck = loadRowPages({
    server: "paged",
    pages: (cursor) => ({ rows: cursor === "same" ? [] : [row("k1")], nextCursor: "same", hasMore: true }),
  });
  assert.deepEqual(await stuck.mod.readAllSidebarRowIds(sidebarRowQuery({})), ["k1"]);
});
