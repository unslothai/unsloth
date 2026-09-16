// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

import { readSrc } from "./helpers/kit.ts";

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

test("editing a connection refreshes the catalog with the saved base URL, not the one it replaced", async () => {
  const dialog = readSrc("features/chat/chat-providers-dialog.tsx");
  const saveAt = dialog.indexOf("async function saveProviderEdits()");
  const save = dialog.slice(saveAt, dialog.indexOf("\n  async function ", saveAt + 1));
  const refreshed = save.match(/refreshProviderModelCatalogs\(\[(\w+)\]\)/)?.[1];
  assert.ok(saveAt > 0 && refreshed, "the edit save path no longer refreshes the catalog");
  const literalAt = save.indexOf(`const ${refreshed}: ExternalProviderConfig = {`);
  assert.ok(literalAt > 0, `${refreshed} is not the edited connection`);
  assert.match(save.slice(literalAt, save.indexOf("\n      };", literalAt)), /baseUrl: updated\.base_url/);

  const { refreshProviderModelCatalogs } = await vite.ssrLoadModule(
    "/src/features/chat/sync-external-providers.ts",
  );
  const catalog = await vite.ssrLoadModule("/src/features/chat/model-catalog.ts");
  const asked: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    asked.push(JSON.parse(String(init?.body)).provider_id);
    return Response.json([{ id: "acme/fresh", reasoning: { supported_efforts: ["low", "high"] } }]);
  }) as typeof fetch;
  try {
    await refreshProviderModelCatalogs([
      { id: "edited", providerType: "openrouter", baseUrl: "https://openrouter.ai/api/v1" },
    ]);
    assert.deepEqual(asked, ["edited"]);
  } finally {
    globalThis.fetch = realFetch;
    catalog.clearProviderModelCatalog("openrouter");
  }
});
