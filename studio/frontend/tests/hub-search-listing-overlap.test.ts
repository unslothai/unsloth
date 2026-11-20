// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { mergeTaskIterators } from "../src/features/hub/lib/merge-task-iterators.ts";
import * as filters from "../src/features/hub/lib/model-search-filters.ts";
import { LruMap } from "../src/features/hub/lib/lru-map.ts";
import * as searchText from "../src/features/hub/lib/search-text.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import type * as SearchModule from "../src/features/hub/hooks/use-hub-model-search.ts";

type Options = NonNullable<Parameters<typeof SearchModule.useHubModelSearch>[1]>;

interface HubRequest {
  url: URL;
  signal: AbortSignal | undefined;
  release: (failure?: number) => void;
  settled: boolean;
}

const UNSLOTH = ["unsloth/Coder-A", "unsloth/Coder-B", "unsloth/Coder-C", "unsloth/Coder-D"];
const GENERAL = ["org/Coder-1", "unsloth/Coder-B", "org/Coder-2", "unsloth/Coder-D", "org/Coder-3"];
const PRIORITY = ["unsloth/Pinned-1", "unsloth/Pinned-2"];

/** Hub listings held open until the test releases them, so ordering is observable. */
function fakeHub() {
  const requests: HubRequest[] = [];
  const unhandled: unknown[] = [];
  const fetchWithTimeout = (input: string, init?: RequestInit) =>
    new Promise<Response>((resolve, reject) => {
      const signal = init?.signal ?? undefined;
      const url = new URL(input);
      const entry: HubRequest = {
        url,
        signal,
        settled: false,
        release: (failure) => {
          if (entry.settled) return;
          entry.settled = true;
          if (failure) {
            resolve(new Response("{}", { status: failure }));
            return;
          }
          const author = url.searchParams.get("author");
          const names = url.pathname.startsWith("/api/models/")
            ? [url.pathname.slice("/api/models/".length)]
            : author === "unsloth"
              ? url.searchParams.get("search")
                ? UNSLOTH
                : ["unsloth/Listed-1", "unsloth/Pinned-2", "unsloth/Listed-2"]
              : GENERAL;
          const limit = Number(url.searchParams.get("limit") ?? 500);
          const rows = names.slice(0, limit).map((id) => ({ id, pipeline_tag: "text-generation" }));
          resolve(Response.json(url.pathname.startsWith("/api/models/") ? rows[0] : rows));
        },
      };
      if (signal?.aborted) {
        reject(signal.reason);
        return;
      }
      signal?.addEventListener("abort", () => {
        entry.settled = true;
        reject(signal.reason);
      });
      requests.push(entry);
    });
  // The SDK's listModels, reduced to what the search reads: one page, `limit` honoured.
  async function* listModels(params: {
    hubUrl: string;
    search: { owner?: string; query?: string; tags?: string[] };
    limit?: number;
    sort?: string;
    fetch: typeof fetch;
  }) {
    const url = new URL(`${params.hubUrl}/api/models`);
    url.searchParams.set("limit", String(Math.min(params.limit ?? Infinity, 500)));
    if (params.search.owner) url.searchParams.set("author", params.search.owner);
    if (params.search.query) url.searchParams.set("search", params.search.query);
    for (const tag of params.search.tags ?? []) url.searchParams.append("filter", tag);
    if (params.sort) url.searchParams.set("sort", params.sort);
    const res = await params.fetch(url.toString(), {});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    let left = params.limit ?? Infinity;
    for (const item of (await res.json()) as { id: string; pipeline_tag: string }[]) {
      yield { name: item.id, task: item.pipeline_tag };
      if (--left <= 0) return;
    }
  }
  const modelInfo = async (params: { hubUrl: string; name: string; fetch: typeof fetch }) => {
    const res = await params.fetch(`${params.hubUrl}/api/models/${params.name}`, {});
    const item = (await res.json()) as { id: string; pipeline_tag: string };
    return { name: item.id, pipeline_tag: item.pipeline_tag };
  };
  const onUnhandled = (reason: unknown) => unhandled.push(reason);
  process.on("unhandledRejection", onUnhandled);
  return {
    requests,
    unhandled,
    listModels,
    modelInfo,
    fetchWithTimeout,
    dispose: () => process.off("unhandledRejection", onUnhandled),
  };
}

function load(hub: ReturnType<typeof fakeHub>) {
  let iterator!: AsyncGenerator<unknown>;
  const memo = (create: () => unknown) => create();
  const mod = loadWithStubs<typeof SearchModule>(
    new URL("../src/features/hub/hooks/use-hub-model-search.ts", import.meta.url),
    {
      react: {
        useMemo: memo,
        useCallback: (fn: unknown) => fn,
        useState: (value: unknown) => [value, () => {}],
        useEffect: () => {},
        startTransition: (fn: () => void) => fn(),
      },
      "@/config/env": {
        usePlatformStore: (select: (s: unknown) => unknown) => select({ deviceType: "linux" }),
      },
      "@/lib/model-size": { ggufRepoParams: () => undefined },
      "@/lib/hf-endpoint": {
        getHfEndpoint: () => "https://huggingface.co",
        useHfEndpoint: () => "https://huggingface.co",
      },
      "@huggingface/hub": { listModels: hub.listModels, modelInfo: hub.modelInfo },
      "../lib/hf-cache": {
        // The real cache runs three lookups at a time; one at a time here keeps the test simple.
        cachedModelInfo: hub.modelInfo,
        primeCacheFromListing: () => {},
      },
      "../lib/hf-model-meta": {
        EMBEDDING_TAGS: new Set(),
        estimateSizeFromDtypes: () => undefined,
      },
      "../lib/merge-task-iterators": { mergeTaskIterators },
      "../lib/model-capabilities": { detectBaseModel: () => undefined },
      "../lib/model-identifiers": { isGgufLike: () => false },
      "../lib/fresh-ranking": { rankFresh: (models: unknown[]) => models },
      "../lib/provider-logos": {},
      "../lib/provider-search": { matchProviderByQuery: () => null },
      "../lib/network": { fetchWithTimeout: hub.fetchWithTimeout },
      "../lib/unsloth-support": {
        classifyUnslothSupport: () => ({ status: "supported" }),
      },
      "../lib/model-search-filters": filters,
      "../lib/lru-map": { LruMap },
      "../lib/search-text": searchText,
      "./use-hub-paginated-search": {
        MAX_RAW_ITEMS_PER_BATCH: 192,
        pullBatch: async (iter: AsyncGenerator<unknown>, map: (raw: unknown) => unknown, size: number) => {
          const items: unknown[] = [];
          let scanned = 0;
          while (items.length < size && scanned < 192) {
            const next = await iter.next();
            if (next.done) return { items, done: true, scanned };
            scanned++;
            const mapped = map(next.value);
            if (mapped) items.push(mapped);
          }
          return { items, done: false, scanned };
        },
        useHubPaginatedSearch: (create: (signal: AbortSignal) => AsyncGenerator<unknown>) => {
          iterator = create(controller.signal);
          return { results: [] };
        },
      },
    },
  );
  let controller = new AbortController();
  return {
    mod,
    controller: () => controller,
    open(query: string, options: Options = {}) {
      controller = new AbortController();
      mod.useHubModelSearch(query, options);
      return iterator;
    },
  };
}

const ticks = async (n = 20) => {
  for (let i = 0; i < n; i++) await new Promise((resolve) => setImmediate(resolve));
};
const nameOf = (row: unknown) => (row as { name: string }).name;
const kind = (request: HubRequest) =>
  request.url.pathname.startsWith("/api/models/")
    ? "info"
    : request.url.searchParams.get("author") === "unsloth"
      ? "unsloth"
      : "general";

test("a typed search requests the general listing before the Unsloth one answers", { timeout: 5_000 }, async (t) => {
  const hub = fakeHub();
  t.after(hub.dispose);
  const iter = load(hub).open("coder");
  const first = iter.next();
  await ticks();
  assert.deepEqual(hub.requests.map(kind), ["unsloth", "general"]);
  assert.equal(hub.requests[0].settled, false);
  hub.requests[1].release();
  await ticks();
  hub.requests[0].release();
  const rows = [nameOf((await first).value)];
  for await (const row of iter) rows.push(nameOf(row));
  assert.deepEqual(rows, [
    "unsloth/Coder-A",
    "unsloth/Coder-B",
    "unsloth/Coder-C",
    "org/Coder-1",
    "org/Coder-2",
    "unsloth/Coder-D",
    "org/Coder-3",
  ]);
  assert.equal(hub.requests.length, 2);
  assert.deepEqual(hub.unhandled, []);
});

test("the Unsloth pass asks the Hub for the three rows it reads, unless a filter is active", { timeout: 5_000 }, async (t) => {
  const hub = fakeHub();
  t.after(hub.dispose);
  const search = load(hub);
  const plain = search.open("coder").next();
  await ticks();
  assert.equal(hub.requests[0].url.searchParams.get("limit"), "3");
  assert.equal(hub.requests[1].url.searchParams.get("limit"), "500");
  search.controller().abort();
  await assert.rejects(plain, { name: "AbortError" });

  hub.requests.length = 0;
  const filtered = search
    .open("coder", { filters: { minContext: 2048 } as filters.ModelSearchFilters })
    .next();
  await ticks();
  const unsloth = hub.requests.find((r) => kind(r) === "unsloth");
  assert.equal(unsloth?.url.searchParams.get("limit"), "500");
  search.controller().abort();
  await assert.rejects(filtered, { name: "AbortError" });
  await ticks();
  assert.deepEqual(hub.unhandled, []);
});

test("a new keystroke aborts both listings and leaves no unhandled rejection", { timeout: 5_000 }, async (t) => {
  const hub = fakeHub();
  t.after(hub.dispose);
  const search = load(hub);
  const iter = search.open("coder");
  const first = iter.next();
  await ticks();
  assert.equal(hub.requests.length, 2);
  search.controller().abort();
  await assert.rejects(first, { name: "AbortError" });
  await ticks();
  assert.ok(hub.requests.every((r) => r.signal?.aborted));
  assert.deepEqual(hub.unhandled, []);
});

test("a failed Unsloth listing surfaces its own error and cancels the general one", { timeout: 5_000 }, async (t) => {
  const hub = fakeHub();
  t.after(hub.dispose);
  const iter = load(hub).open("coder");
  const first = iter.next();
  await ticks();
  hub.requests[0].release(500);
  await assert.rejects(first, /HTTP 500/);
  await ticks();
  assert.equal(hub.requests[1].signal?.aborted, true);
  assert.deepEqual(hub.unhandled, []);
});

test("a failed general listing surfaces after the Unsloth rows, where it did before", { timeout: 5_000 }, async (t) => {
  const hub = fakeHub();
  t.after(hub.dispose);
  const iter = load(hub).open("coder");
  const first = iter.next();
  await ticks();
  const general = hub.requests.find((r) => kind(r) === "general");
  general?.release(500);
  await ticks();
  hub.requests[0].release();
  const rows = [nameOf((await first).value)];
  await assert.rejects(async () => {
    for await (const row of iter) rows.push(nameOf(row));
  }, /HTTP 500/);
  assert.deepEqual(rows, ["unsloth/Coder-A", "unsloth/Coder-B", "unsloth/Coder-C"]);
  assert.deepEqual(hub.unhandled, []);
});

test("the training picker loads its listing while the priority lookups finish", { timeout: 5_000 }, async (t) => {
  const hub = fakeHub();
  t.after(hub.dispose);
  const iter = load(hub).open("", { priorityIds: PRIORITY });
  const first = iter.next();
  await ticks();
  assert.deepEqual(hub.requests.map(kind), ["info", "info", "unsloth"]);
  for (const request of hub.requests) request.release();
  const rows = [nameOf((await first).value)];
  for await (const row of iter) rows.push(nameOf(row));
  assert.deepEqual(rows, [
    "unsloth/Pinned-1",
    "unsloth/Pinned-2",
    "unsloth/Listed-1",
    "unsloth/Listed-2",
  ]);
  assert.deepEqual(hub.unhandled, []);
});

test("channel first pages and the fresh trending pool ask only for the rows they read", { timeout: 5_000 }, async (t) => {
  const hub = fakeHub();
  t.after(hub.dispose);
  const { mod } = load(hub);
  const page = (sortBy: SearchModule.HfSortKey) =>
    mod.fetchChannelFirstPage({
      channel: { owner: "unsloth" },
      sortBy,
      signal: new AbortController().signal,
      deviceType: "linux",
    });
  const trending = page("trendingScore");
  await ticks();
  assert.equal(hub.requests[0].url.searchParams.get("limit"), "192");
  hub.requests[0].release();
  await trending;

  hub.requests.length = 0;
  const fresh = page("fresh");
  await ticks();
  const limits = Object.fromEntries(
    hub.requests.map((r) => [r.url.searchParams.get("sort"), r.url.searchParams.get("limit")]),
  );
  // The newest listing is read past its pool of 100; the trending one never is.
  assert.deepEqual(limits, { createdAt: "500", trendingScore: "100" });
  for (const request of hub.requests) request.release();
  await fresh;
  assert.deepEqual(hub.unhandled, []);
});
