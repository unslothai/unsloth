// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { listModels, modelInfo } from "@huggingface/hub";
import type * as SearchModule from "../src/features/hub/hooks/use-hub-model-search.ts";
import { LruMap } from "../src/features/hub/lib/lru-map.ts";
import { mergeTaskIterators } from "../src/features/hub/lib/merge-task-iterators.ts";
import * as filters from "../src/features/hub/lib/model-search-filters.ts";
import * as searchText from "../src/features/hub/lib/search-text.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Options = NonNullable<
  Parameters<typeof SearchModule.useHubModelSearch>[1]
>;

const range = (n: number, name: (i: number) => string) =>
  Array.from({ length: n }, (_, i) => name(i));

/** Every listing the tests read, longer than one 500-row page where it matters. */
function rowsFor(url: URL): string[] {
  const author = url.searchParams.get("author");
  const search = url.searchParams.get("search");
  const sort = url.searchParams.get("sort");
  if (author === "unsloth" && search === "coder") {
    return [
      "unsloth/Coder-A",
      "unsloth/Coder-B",
      "unsloth/Coder-C",
      "unsloth/Coder-D",
    ];
  }
  if (search === "coder") {
    return range(700, (i) =>
      i === 1
        ? "unsloth/Coder-B"
        : i === 250
          ? "unsloth/Coder-D"
          : `org/Coder-${i}`,
    );
  }
  if (author === "unsloth" && search === "qwen") {
    // Every third is another maker's distill: kept out of the maker's own group.
    return range(260, (i) =>
      i % 3 === 2 ? `unsloth/DeepSeek-Distill-Qwen-${i}` : `unsloth/Qwen-${i}`,
    );
  }
  if (author === "Qwen") return range(230, (i) => `Qwen/Qwen-${i}`);
  if (search === "qwen")
    return range(40, (i) => (i === 3 ? "Qwen/Qwen-3" : `org/qwen-${i}`));
  if (author === "unsloth") return range(600, (i) => `unsloth/${sort}-${i}`);
  return range(600, (i) => `any/${sort}-${i}`);
}

/**
 * A Hub that pages like the real one: `limit` rows from the cursor, and a next link that repeats
 * the request's limit. `pageSize` overrides the rows served per page.
 */
function fakeHub(pageSize?: (requested: number) => number) {
  const requests: URL[] = [];
  const fetchWithTimeout = async (input: string) => {
    const url = new URL(input);
    requests.push(url);
    const names = rowsFor(url);
    const offset = Number(url.searchParams.get("cursor") ?? 0);
    const requested = Number(url.searchParams.get("limit") ?? 500);
    const size = pageSize ? pageSize(requested) : requested;
    const rows = names.slice(offset, offset + size).map((id, i) => ({
      _id: id,
      id,
      pipeline_tag: "text-generation",
      downloads: names.length - offset - i,
      lastModified: "2026-01-01T00:00:00.000Z",
    }));
    const headers = new Headers();
    if (offset + size < names.length) {
      const next = new URL(url);
      next.searchParams.set("cursor", String(offset + size));
      headers.set("Link", `<${next}>; rel="next"`);
    }
    return Response.json(rows, { headers });
  };
  return { requests, fetchWithTimeout };
}

const QWEN = { id: "qwen" };

function load(hub: ReturnType<typeof fakeHub>) {
  let iterator!: AsyncGenerator<unknown>;
  const mod = loadWithStubs<typeof SearchModule>(
    new URL(
      "../src/features/hub/hooks/use-hub-model-search.ts",
      import.meta.url,
    ),
    {
      react: {
        useMemo: (create: () => unknown) => create(),
        useCallback: (fn: unknown) => fn,
        useState: (value: unknown) => [value, () => {}],
        useEffect: () => {},
        startTransition: (fn: () => void) => fn(),
      },
      "@/config/env": {
        usePlatformStore: (select: (s: unknown) => unknown) =>
          select({ deviceType: "linux" }),
      },
      "@/lib/model-size": { ggufRepoParams: () => undefined },
      "@/lib/hf-endpoint": {
        getHfEndpoint: () => "https://huggingface.co",
        useHfEndpoint: () => "https://huggingface.co",
      },
      // The real SDK, so its own limit and next-link handling is what runs.
      "@huggingface/hub": { listModels, modelInfo },
      "../lib/hf-cache": {
        cachedModelInfo: modelInfo,
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
      "../lib/provider-logos": {
        resolveOwnerProviderLogo: (_owner: string, repo = "") =>
          repo.toLowerCase().startsWith("qwen") ? QWEN : null,
      },
      "../lib/provider-search": {
        matchProviderByQuery: (query: string) =>
          query === "qwen" ? QWEN : null,
        providerSearchTerms: () => ["qwen"],
        providerHubOrgs: () => ["Qwen"],
      },
      "../lib/network": { fetchWithTimeout: hub.fetchWithTimeout },
      "../lib/unsloth-support": {
        classifyUnslothSupport: () => ({ status: "supported" }),
      },
      "../lib/model-search-filters": filters,
      "../lib/lru-map": { LruMap },
      "../lib/search-text": searchText,
      "./use-hub-paginated-search": {
        MAX_RAW_ITEMS_PER_BATCH: 192,
        pullBatch: async (
          iter: AsyncGenerator<unknown>,
          map: (raw: unknown) => unknown,
          size: number,
        ) => {
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
        useHubPaginatedSearch: (
          create: (signal: AbortSignal) => AsyncGenerator<unknown>,
        ) => {
          iterator = create(new AbortController().signal);
          return { results: [] };
        },
      },
    },
  );
  return {
    mod,
    open(query: string, options: Options = {}) {
      mod.useHubModelSearch(query, options);
      return iterator;
    },
  };
}

async function drain(iter: AsyncGenerator<unknown>): Promise<string[]> {
  const rows: string[] = [];
  for await (const row of iter) rows.push((row as { name: string }).name);
  return rows;
}

/** The rows and order a Hub serving full 500-row pages gives: what the search read before. */
async function uncapped(query: string, options: Options = {}) {
  const hub = fakeHub(() => 500);
  return drain(load(hub).open(query, options));
}

const listing = (url: URL) =>
  [
    url.searchParams.get("author") ?? "",
    url.searchParams.get("search") ?? "",
  ].join(":");
const limits = (requests: URL[], key: string) =>
  requests
    .filter((url) => listing(url) === key)
    .map(
      (url) =>
        `${url.searchParams.has("cursor") ? "next" : "first"}=${url.searchParams.get("limit")}`,
    );

test("a typed search asks for 192 general rows first, then pages on at 500 with the same order", async () => {
  const hub = fakeHub();
  const rows = await drain(load(hub).open("coder"));
  assert.deepEqual(limits(hub.requests, "unsloth:coder"), ["first=3"]);
  assert.deepEqual(limits(hub.requests, ":coder"), [
    "first=192",
    "next=500",
    "next=500",
  ]);
  assert.deepEqual(rows, await uncapped("coder"));
  // Unsloth-B is the general listing's only repeat.
  assert.equal(rows.length, 3 + 699);
  // The next page starts on the row after the first page's last.
  const general = rows.slice(3);
  assert.equal(general[190], "org/Coder-191");
  assert.equal(general[191], "org/Coder-192");
});

test("a model maker's listings ask for their pool first and continue past it unchanged", async () => {
  const hub = fakeHub();
  const rows = await drain(load(hub).open("qwen"));
  assert.deepEqual(limits(hub.requests, "unsloth:qwen").slice(0, 2), [
    "first=100",
    "next=500",
  ]);
  assert.deepEqual(limits(hub.requests, "Qwen:"), ["first=100", "next=500"]);
  // The search's own listings (Unsloth pass, then the general one) are read after the maker groups.
  assert.ok(limits(hub.requests, "unsloth:qwen").includes("first=3"));
  assert.deepEqual(limits(hub.requests, ":qwen"), ["first=192"]);
  assert.deepEqual(rows, await uncapped("qwen"));
  assert.equal(new Set(rows).size, rows.length);
});

test("the Unsloth-only typed search and a channel listing ask for 192 rows first", async () => {
  const hub = fakeHub();
  const search = load(hub);
  const org = await drain(search.open("coder", { ownerScope: "unsloth" }));
  assert.deepEqual(limits(hub.requests, "unsloth:coder"), ["first=192"]);
  assert.deepEqual(org, await uncapped("coder", { ownerScope: "unsloth" }));

  hub.requests.length = 0;
  const channel = { owner: "unsloth" };
  const rows = await drain(search.open("", { channel, sortBy: "likes" }));
  assert.deepEqual(limits(hub.requests, "unsloth:"), ["first=192", "next=500"]);
  assert.deepEqual(rows, await uncapped("", { channel, sortBy: "likes" }));
  assert.equal(rows.length, 600);
});

test("listings that already set their size keep it", async () => {
  const hub = fakeHub();
  const { mod } = load(hub);
  const first = await mod.fetchChannelFirstPage({
    channel: { owner: "unsloth" },
    sortBy: "trendingScore",
    signal: new AbortController().signal,
    deviceType: "linux",
    pageSize: 1000,
  });
  // An SDK limit ends the listing: one request, never a next page.
  assert.deepEqual(limits(hub.requests, "unsloth:"), ["first=192"]);
  assert.equal(first.scanned, 192);

  hub.requests.length = 0;
  await mod.fetchChannelFirstPage({
    channel: { owner: "unsloth" },
    sortBy: "fresh",
    signal: new AbortController().signal,
    deviceType: "linux",
  });
  const bySort = (sort: string) =>
    hub.requests
      .filter((url) => url.searchParams.get("sort") === sort)
      .map((url) => url.searchParams.get("limit"));
  assert.deepEqual(bySort("trendingScore"), ["100"]);
  // The newest listing is read past its pool of 100: its first page holds a whole batch.
  assert.deepEqual(bySort("createdAt"), ["192"]);

  // A plain browse with no query and no channel is unchanged.
  hub.requests.length = 0;
  const browse = load(hub).open("", { sortBy: "likes" });
  await browse.next();
  await browse.return(undefined);
  assert.deepEqual(limits(hub.requests, ":"), ["first=500"]);
});
