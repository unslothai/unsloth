// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as filters from "../src/features/hub/lib/model-search-filters.ts";
import { LruMap } from "../src/features/hub/lib/lru-map.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import type * as SearchModule from "../src/features/hub/hooks/use-hub-model-search.ts";

type Options = NonNullable<
  Parameters<typeof SearchModule.useHubModelSearch>[1]
>;

function searchHarness(request: typeof fetch) {
  let endpoint = "https://huggingface.co";
  let index = 0;
  const cells: { deps: unknown[]; value: unknown }[] = [];
  const memo = (create: () => unknown, deps: unknown[]) => {
    const i = index++;
    const cell = cells[i];
    if (
      !cell ||
      deps.length !== cell.deps.length ||
      deps.some((dep, j) => !Object.is(dep, cell.deps[j]))
    ) {
      cells[i] = { deps, value: create() };
    }
    return cells[i].value;
  };
  let iterator!: AsyncGenerator<unknown>;
  const rows = [
    { name: "unsloth/small", safetensors: { total: 4e9 } },
    { name: "unsloth/large", safetensors: { total: 8e9 } },
  ];
  const mod = loadWithStubs<typeof SearchModule>(
    new URL(
      "../src/features/hub/hooks/use-hub-model-search.ts",
      import.meta.url,
    ),
    {
      react: {
        useMemo: memo,
        useCallback: (fn: unknown, deps: unknown[]) => memo(() => fn, deps),
        useState: (value: unknown) => [value, () => {}],
        useEffect: () => {},
      },
      "@/config/env": {
        usePlatformStore: (select: (s: unknown) => unknown) =>
          select({ deviceType: "linux" }),
      },
      "@/lib/model-size": {
        ggufRepoParams: (_id: string, total: number | undefined) => total,
      },
      "@/lib/hf-endpoint": {
        getHfEndpoint: () => endpoint,
        useHfEndpoint: () => endpoint,
      },
      "@huggingface/hub": {
        listModels: async function* () {
          yield* rows;
        },
      },
      "../lib/hf-cache": {},
      "../lib/hf-model-meta": {
        EMBEDDING_TAGS: new Set(),
        estimateSizeFromDtypes: () => undefined,
      },
      "../lib/merge-task-iterators": {
        mergeTaskIterators: (
          _tasks: unknown,
          create: (
            task: undefined,
            signal?: AbortSignal,
          ) => AsyncGenerator<unknown>,
          signal?: AbortSignal,
        ) => create(undefined, signal),
      },
      "../lib/model-capabilities": { detectBaseModel: () => undefined },
      "../lib/model-identifiers": { isGgufLike: () => false },
      "../lib/fresh-ranking": {},
      "../lib/provider-logos": {},
      "../lib/provider-search": {},
      "../lib/network": { fetchWithTimeout: request },
      "../lib/unsloth-support": {
        classifyUnslothSupport: () => ({ status: "supported" }),
      },
      "../lib/model-search-filters": filters,
      "../lib/lru-map": { LruMap },
      "./use-hub-paginated-search": {
        useHubPaginatedSearch: (
          create: (signal: AbortSignal) => AsyncGenerator<unknown>,
        ) => {
          iterator = create(controller.signal);
          return { results: [] };
        },
      },
    },
  );
  let controller = new AbortController();
  return {
    setEndpoint: (value: string) => {
      endpoint = value;
    },
    abort: () => controller.abort(),
    async search(next: filters.ModelSearchFilters, options: Options = {}) {
      controller.abort();
      controller = new AbortController();
      index = 0;
      mod.useHubModelSearch("", {
        ownerScope: "unsloth",
        filters: next,
        ...options,
      });
      const found: unknown[] = [];
      for await (const row of iterator) if (row) found.push(row);
      return found;
    },
    rows,
  };
}

test("changing context bounds reuses configs instead of exhausting the Hub quota", async () => {
  let requests = 0;
  const hook = searchHarness(async (url) => {
    if (++requests > 2) return new Response(null, { status: 429 });
    return Response.json({
      max_position_embeddings: String(url).includes("/small/") ? 4096 : 32768,
    });
  });
  assert.deepEqual(await hook.search({ minContext: 2048 }), hook.rows);
  assert.deepEqual(await hook.search({ minContext: 8192 }), [hook.rows[1]]);
  assert.deepEqual(await hook.search({}), hook.rows);
  assert.deepEqual(await hook.search({ maxContext: 8192 }), [hook.rows[0]]);
  assert.equal(requests, 2);
});

test("a retry reuses successful configs but fetches the failed request again", async () => {
  const requests = new Map<string, number>();
  const hook = searchHarness(async (url) => {
    const key = String(url);
    const count = (requests.get(key) ?? 0) + 1;
    requests.set(key, count);
    if (key.includes("/large/") && count === 1)
      return new Response(null, { status: 429 });
    return Response.json({ max_position_embeddings: 32768 });
  });
  await assert.rejects(hook.search({ minContext: 2048 }), /429/);
  assert.deepEqual(await hook.search({ minContext: 2048 }), hook.rows);
  assert.equal(
    requests.get(
      "https://huggingface.co/unsloth/small/resolve/main/config.json",
    ),
    1,
  );
  assert.equal(
    requests.get(
      "https://huggingface.co/unsloth/large/resolve/main/config.json",
    ),
    2,
  );
});

test("a different token or Hub endpoint cannot reuse filter metadata", async () => {
  let requests = 0;
  const hook = searchHarness(async () => {
    requests++;
    return Response.json({ max_position_embeddings: 32768 });
  });
  for (const accessToken of [
    undefined,
    "first-token",
    "second-token",
    undefined,
  ]) {
    await hook.search({ minContext: 2048 }, { accessToken });
  }
  hook.setEndpoint("https://mirror.example");
  await hook.search({ minContext: 2048 });
  assert.equal(requests, 10);
});

test("expired metadata is refreshed so a changed context limit takes effect", async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: 1_000_000 });
  let context = 32768;
  let requests = 0;
  const hook = searchHarness(async () => {
    requests++;
    return Response.json({ max_position_embeddings: context });
  });
  await hook.search({ minContext: 16384 });
  context = 8192;
  t.mock.timers.tick(60_000);
  assert.deepEqual(await hook.search({ minContext: 16384 }), hook.rows);
  assert.equal(requests, 2);
  t.mock.timers.tick(5 * 60_000);
  assert.deepEqual(await hook.search({ minContext: 16384 }), []);
  assert.equal(requests, 4);
});

test("missing configs are reused but access errors and malformed JSON are retried", async () => {
  for (const status of [404, 401, 403, 200]) {
    let requests = 0;
    const hook = searchHarness(async () => {
      requests++;
      return new Response("not JSON", { status });
    });
    assert.deepEqual(await hook.search({ minContext: 2048 }), []);
    assert.deepEqual(await hook.search({ minContext: 8192 }), []);
    assert.equal(requests, status === 404 ? 2 : 4, `HTTP ${status}`);
  }
});

test("an aborted search cannot cache metadata that arrives after cancellation", async () => {
  let release!: () => void;
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  let requests = 0;
  const hook = searchHarness(async () => {
    requests++;
    await pending;
    return Response.json({ max_position_embeddings: 32768 });
  });
  const first = hook.search({ minContext: 2048 });
  await new Promise((resolve) => setImmediate(resolve));
  hook.abort();
  release();
  await first;
  assert.deepEqual(await hook.search({ minContext: 8192 }), hook.rows);
  assert.equal(requests, 4);
});

test("rows the list hides cost no metadata request", async () => {
  const requested: string[] = [];
  const hook = searchHarness(async (url) => {
    requested.push(String(url));
    return Response.json({ max_position_embeddings: 32768 });
  });
  const keepResult = (result: { id: string }) => result.id !== "unsloth/small";
  assert.deepEqual(await hook.search({ minContext: 2048 }, { keepResult }), [
    hook.rows[1],
  ]);
  assert.deepEqual(requested, [
    "https://huggingface.co/unsloth/large/resolve/main/config.json",
  ]);
  // Without a per-row lookup nothing is skipped: the list keeps filtering as before.
  assert.deepEqual(
    await hook.search({ maxParams: 9e9 }, { keepResult: () => false }),
    hook.rows,
  );
  assert.equal(requested.length, 1);
});

test("a row the mapper throws on is skipped, not the whole search", async () => {
  const hook = searchHarness(async () =>
    Response.json({ max_position_embeddings: 32768 }),
  );
  const keepResult = (result: { id: string }) => {
    if (result.id === "unsloth/small") throw new RangeError("Invalid time value");
    return true;
  };
  assert.deepEqual(await hook.search({ minContext: 2048 }, { keepResult }), [
    hook.rows[1],
  ]);
});
