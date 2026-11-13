// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

let vite: ViteDevServer;
let liveModelSeeds: (seeds: string[], live: string[]) => string[];

before(async () => {
  vite = await createServer({ appType: "custom", server: { middlewareMode: true } });
  const loaded = await vite.ssrLoadModule("/src/features/chat/chat-providers-dialog.tsx");
  liveModelSeeds = loaded.liveModelSeeds;
});

after(async () => {
  await vite.close();
});

test("a provider's own list drops suggestions it did not return", () => {
  assert.deepEqual(
    liveModelSeeds(["gpt-9", "gpt-8", "gpt-7"], ["gpt-7", "gpt-9", "ft-private"]),
    ["gpt-9", "gpt-7"],
  );
});

test("suggestions stand in until a provider has listed its models", () => {
  assert.deepEqual(liveModelSeeds(["gpt-9", "gpt-8"], []), ["gpt-9", "gpt-8"]);
});

test("one stale catalog is fetched once, however many callers ask", async () => {
  const { refreshModelsDevCatalog } = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  const catalog = await vite.ssrLoadModule("/src/features/chat/model-catalog.ts");
  let fetches = 0;
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    fetches += 1;
    return Response.json({ fetched_at: Date.now() / 1000, providers: {} });
  }) as typeof fetch;
  try {
    assert.deepEqual(
      await Promise.all([refreshModelsDevCatalog(), refreshModelsDevCatalog()]),
      [true, true],
    );
    assert.equal(fetches, 1);
    assert.equal(await refreshModelsDevCatalog(), false);
    assert.equal(fetches, 1);
    assert.equal(typeof catalog.modelsDevCatalogFetchedAt(), "number");
  } finally {
    globalThis.fetch = realFetch;
  }
});
