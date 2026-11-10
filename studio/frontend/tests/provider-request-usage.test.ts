// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";
import { createServer } from "vite";
import type { ExternalProviderConfig } from "../src/features/chat/external-providers.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
Object.assign(window.location, { href: "http://localhost/" });
const vite = await createServer({
  appType: "custom",
  logLevel: "silent",
  optimizeDeps: { noDiscovery: true },
  server: { middlewareMode: true, watch: null },
});
after(() => vite.close());
const providers = await vite.ssrLoadModule(
  "/src/features/chat/external-providers.ts",
);
const api = await vite.ssrLoadModule("/src/features/chat/api/providers-api.ts");
const sync = await vite.ssrLoadModule(
  "/src/features/chat/sync-external-providers.ts",
);
const routing = await vite.ssrLoadModule(
  "/src/features/chat/utils/chat-title.ts",
);

const connection: ExternalProviderConfig = {
  id: "custom-a",
  providerType: "custom",
  backendProviderType: "custom",
  name: "Gateway",
  baseUrl: "http://localhost:9000/v1",
  models: ["m"],
  createdAt: 1,
  updatedAt: 1,
};

async function routingFields(provider: ExternalProviderConfig) {
  return routing.buildExternalRoutingFields({ provider, modelId: "m", apiKey: "" });
}

test("the switch maps to request_usage on create and update, and only when set", async () => {
  const bodies: Record<string, unknown>[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (_url, init) => {
    bodies.push(JSON.parse(String(init?.body)));
    return Response.json({ request_usage: true });
  }) as typeof fetch;
  try {
    const created = await api.createProviderConfig({
      providerType: "custom",
      displayName: "Gateway",
      requestUsage: true,
    });
    assert.equal(created.request_usage, true);
    await api.createProviderConfig({ providerType: "custom", displayName: "Gateway" });
    await api.updateProviderConfig("custom-a", { requestUsage: false });
    await api.updateProviderConfig("custom-a", { displayName: "Renamed" });
  } finally {
    globalThis.fetch = original;
  }
  assert.deepEqual(
    bodies.map((body) => body.request_usage),
    [true, undefined, false, undefined],
  );
  assert.equal("request_usage" in bodies[3], false);
});

test("only a Custom connection that opted in sends provider_request_usage", async () => {
  providers.saveExternalProviders([
    { ...connection, requestUsage: true },
    { ...connection, id: "custom-b" },
    { ...connection, id: "vllm-a", providerType: "vllm", requestUsage: true },
  ]);
  const [on, off, listed] = providers.loadExternalProviders();
  assert.equal(on.requestUsage, true);
  assert.equal(off.requestUsage, false);
  assert.equal(listed.requestUsage, undefined);
  assert.equal((await routingFields(on)).provider_request_usage, true);
  assert.equal("provider_request_usage" in (await routingFields(off)), false);
  assert.equal("provider_request_usage" in (await routingFields(listed)), false);
});

test("server sync takes the server's request_usage over the local copy", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = (async (url) => {
    const path = String(url);
    if (path.includes("/registry")) return Response.json([]);
    if (path.endsWith("/api/providers/"))
      return Response.json(
        ["custom-a", "custom-b"].map((id, index) => ({
          id,
          provider_type: "custom",
          display_name: "Gateway",
          base_url: connection.baseUrl,
          is_enabled: true,
          models: ["m"],
          available_models: ["m"],
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
          request_usage: index === 1,
        })),
      );
    return Response.json({ fetched_at: 1, providers: {} });
  }) as typeof fetch;
  try {
    const synced = await sync.syncExternalProvidersFromBackend([
      { ...connection, requestUsage: true },
      { ...connection, id: "custom-b", requestUsage: false },
    ]);
    assert.deepEqual(
      synced.map((p: ExternalProviderConfig) => p.requestUsage),
      [false, true],
    );
  } finally {
    globalThis.fetch = original;
  }
});
