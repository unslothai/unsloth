// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

let vite: ViteDevServer;

before(async () => {
  vite = await createServer({ appType: "custom", server: { middlewareMode: true } });
});

after(async () => {
  await vite.close();
});

test("a failing OpenRouter connection leaves the catalog for the next connection to refresh", async () => {
  const { refreshProviderModelCatalogs } = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  const catalog = await vite.ssrLoadModule("/src/features/chat/model-catalog.ts");
  catalog.setModelsDevCatalog({ fetched_at: Date.now() / 1000, providers: {} });
  const asked: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    const { provider_id: providerId } = JSON.parse(String(init?.body));
    asked.push(providerId);
    return providerId === "expired-key"
      ? Response.json({ detail: "Invalid API key" }, { status: 502 })
      : Response.json([{ id: "acme/fresh", reasoning: { supported_efforts: ["low", "high"] } }]);
  }) as typeof fetch;
  try {
    await refreshProviderModelCatalogs([
      { id: "expired-key", providerType: "openrouter" },
      { id: "working-key", providerType: "openrouter" },
      { id: "second-working-key", providerType: "openrouter" },
    ]);
  } finally {
    globalThis.fetch = realFetch;
  }
  assert.deepEqual(asked, ["expired-key", "working-key"]);
  assert.deepEqual(catalog.resolveModelCatalogEntry("openrouter", "acme/fresh")?.efforts, ["low", "high"]);
  catalog.clearProviderModelCatalog("openrouter");
});

test("an OpenRouter connection on a gateway base URL never writes the shared catalog", async () => {
  const { refreshProviderModelCatalogs } = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  const catalog = await vite.ssrLoadModule("/src/features/chat/model-catalog.ts");
  catalog.setModelsDevCatalog({ fetched_at: Date.now() / 1000, providers: {} });
  const asked: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    const { provider_id: providerId } = JSON.parse(String(init?.body));
    asked.push(providerId);
    return providerId === "gateway"
      ? Response.json([{ id: "acme/fresh", reasoning: { supported_efforts: ["max"] } }])
      : Response.json([{ id: "acme/fresh", reasoning: { supported_efforts: ["low", "high"] } }]);
  }) as typeof fetch;
  try {
    await refreshProviderModelCatalogs([
      { id: "gateway", providerType: "openrouter", baseUrl: "https://gateway.example/api/v1" },
      { id: "openrouter", providerType: "openrouter", baseUrl: "https://openrouter.ai/api/v1/" },
    ]);
  } finally {
    globalThis.fetch = realFetch;
  }
  try {
    assert.deepEqual(asked, ["openrouter"]);
    assert.deepEqual(catalog.resolveModelCatalogEntry("openrouter", "acme/fresh")?.efforts, ["low", "high"]);
  } finally {
    catalog.clearProviderModelCatalog("openrouter");
  }
});
