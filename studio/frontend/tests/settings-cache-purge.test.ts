// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A purge names caches by key. The client must never be able to send a path, and a
// bulk clear must never sweep up the caches whose deletion re-downloads models.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type CachesApi = {
  loadCacheInventory: () => Promise<unknown>;
  purgeCaches: (keys: readonly string[]) => Promise<unknown>;
  inventoryFromApi: (value: unknown) => {
    caches: { key: string; sizeBytes: number; optIn: boolean }[];
    totalBytes: number;
    reclaimableBytes: number;
    freeBytes: number | null;
  };
  bulkPurgeKeys: (inventory: unknown) => string[];
};

type Request = { url: string; init: RequestInit | undefined };

function loadApi(responder: (url: string, init?: RequestInit) => Response): {
  api: CachesApi;
  requests: Request[];
} {
  const requests: Request[] = [];
  const api = loadWithStubs<CachesApi>(
    new URL("../src/features/settings/api/caches.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (url: string, init?: RequestInit) => {
          requests.push({ url, init });
          return responder(url, init);
        },
      },
      "@/lib/format-fastapi-error": {
        readFastApiError: async (_response: Response, fallback: string) =>
          fallback,
      },
    },
  );
  return { api, requests };
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

const apiEntry = (
  key: string,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> => ({
  key,
  group: "packages",
  // biome-ignore lint/style/useNamingConvention: API schema
  opt_in: false,
  paths: [`/cache/${key}`],
  // biome-ignore lint/style/useNamingConvention: API schema
  size_bytes: 1000,
  // biome-ignore lint/style/useNamingConvention: API schema
  entry_count: 2,
  present: true,
  purgeable: true,
  // biome-ignore lint/style/useNamingConvention: API schema
  blocked_reason: null,
  ...overrides,
});

const apiInventory = (
  caches: Record<string, unknown>[],
): Record<string, unknown> => ({
  caches,
  // biome-ignore lint/style/useNamingConvention: API schema
  total_bytes: 1000,
  // biome-ignore lint/style/useNamingConvention: API schema
  reclaimable_bytes: 1000,
  // biome-ignore lint/style/useNamingConvention: API schema
  free_bytes: 42,
  // biome-ignore lint/style/useNamingConvention: API schema
  total_disk_bytes: 100,
});

test("a purge posts cache keys and nothing that could be a path", async () => {
  const { api, requests } = loadApi(() =>
    json({
      results: [
        {
          key: "uv",
          // biome-ignore lint/style/useNamingConvention: API schema
          freed_bytes: 5,
          // biome-ignore lint/style/useNamingConvention: API schema
          removed_entries: 1,
          errors: [],
        },
      ],
      // biome-ignore lint/style/useNamingConvention: API schema
      freed_bytes: 5,
      inventory: apiInventory([]),
    }),
  );

  await api.purgeCaches(["uv", "triton"]);

  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "/api/settings/caches/purge");
  assert.equal(requests[0].init?.method, "POST");
  const body = JSON.parse(String(requests[0].init?.body));
  assert.deepEqual(Object.keys(body), ["keys"]);
  assert.deepEqual(body.keys, ["uv", "triton"]);
});

test("the client offers no way to name a directory", () => {
  const source = readFileSync(
    new URL("../src/features/settings/api/caches.ts", import.meta.url),
    "utf8",
  );
  // The request body is built from keys alone: no path, dir or root field.
  assert.match(source, /JSON\.stringify\(\{ keys \}\)/);
  assert.doesNotMatch(source, /path:\s*[^;]*JSON\.stringify/);
});

test("a failed purge surfaces the backend's own message", async () => {
  const { api } = loadApi(() => new Response(null, { status: 400 }));
  await assert.rejects(api.purgeCaches(["uv"]), /Failed to clear the caches/);
});

test("the inventory maps the API's snake_case into the app's shape", async () => {
  const { api } = loadApi(() =>
    json(
      apiInventory([
        apiEntry("uv"),
        // biome-ignore lint/style/useNamingConvention: API schema
        apiEntry("hf_hub", { opt_in: true, group: "models" }),
      ]),
    ),
  );
  const inventory = (await api.loadCacheInventory()) as ReturnType<
    CachesApi["inventoryFromApi"]
  >;
  assert.equal(inventory.caches[0].sizeBytes, 1000);
  assert.equal(inventory.caches[0].optIn, false);
  assert.equal(inventory.caches[1].optIn, true);
  assert.equal(inventory.freeBytes, 42);
});

test("a bulk clear leaves out the model cache, the blocked and the empty", () => {
  const { api } = loadApi(() => json({}));
  const inventory = api.inventoryFromApi(
    apiInventory([
      apiEntry("uv"),
      // Clearing this one re-downloads models, so it is asked for on its own.
      // biome-ignore lint/style/useNamingConvention: API schema
      apiEntry("hf_hub", { opt_in: true }),
      // biome-ignore lint/style/useNamingConvention: API schema
      apiEntry("hf_datasets", { opt_in: true }),
      apiEntry("triton", {
        purgeable: false,
        // biome-ignore lint/style/useNamingConvention: API schema
        blocked_reason: "sits inside the protected folder /outputs",
      }),
      // biome-ignore lint/style/useNamingConvention: API schema
      apiEntry("numba", { present: false, size_bytes: 0, entry_count: 0 }),
      // Nothing in it, so there is nothing for a clear to do.
      // biome-ignore lint/style/useNamingConvention: API schema
      apiEntry("vllm", { size_bytes: 0, entry_count: 0 }),
    ]),
  );
  assert.deepEqual(api.bulkPurgeKeys(inventory), ["uv"]);
});

test("a cache that measures zero bytes but holds entries is still cleared", () => {
  // A tree of empty directories or dangling symlinks costs inodes and directory
  // blocks and the backend can empty it, so a size test would hide the one
  // cache a user cannot easily clear by hand.
  const { api } = loadApi(() => json({}));
  const inventory = api.inventoryFromApi(
    // biome-ignore lint/style/useNamingConvention: API schema
    apiInventory([apiEntry("triton", { size_bytes: 0, entry_count: 40000 })]),
  );
  assert.deepEqual(api.bulkPurgeKeys(inventory), ["triton"]);
});
