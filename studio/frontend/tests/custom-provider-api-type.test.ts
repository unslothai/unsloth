// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";
import { createServer } from "vite";
import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
Object.assign(window.location, { href: "http://localhost/" });
const vite = await createServer({ server: { middlewareMode: true } });
after(() => vite.close());
const api = await vite.ssrLoadModule("/src/features/chat/api/providers-api.ts");
const providers = await vite.ssrLoadModule("/src/features/chat/external-providers.ts");

for (const apiType of ["chat_completions", "responses"] as const) {
  test(`provider create, edit and probe forward ${apiType}`, async () => {
    const bodies: Record<string, unknown>[] = [];
    const previous = globalThis.fetch;
    globalThis.fetch = (async (_url, init) => {
      bodies.push(JSON.parse(String(init?.body)));
      return Response.json({});
    }) as typeof fetch;
    try {
      await api.createProviderConfig({ providerType: "custom", displayName: "Gateway", apiType });
      await api.updateProviderConfig("gateway", { apiType });
      await api.testProviderConnection({ providerType: "custom", apiKey: "", apiType });
    } finally {
      globalThis.fetch = previous;
    }
    assert.equal(bodies.length, 3);
    assert.ok(bodies.every((body) => body.api_type === apiType));
  });
}

test("browser cache preserves Responses and defaults legacy records to Chat Completions", () => {
  const config = {
    id: "gateway", providerType: "custom", name: "Gateway", baseUrl: "https://gateway.example/v1",
    models: ["responses-only"], createdAt: 1, updatedAt: 1,
  };
  providers.saveExternalProviders([{ ...config, apiType: "responses" }]);
  assert.equal(providers.loadExternalProviders()[0].apiType, "responses");
  providers.saveExternalProviders([config]);
  assert.equal(providers.loadExternalProviders()[0].apiType, "chat_completions");
});


test("custom Responses uses the existing Responses sampling controls", async () => {
  const { getProviderCapabilities } = await vite.ssrLoadModule("/src/features/chat/provider-capabilities.ts");
  assert.deepEqual(getProviderCapabilities("custom", "responses"), getProviderCapabilities("openai"));
  assert.equal(getProviderCapabilities("custom", "chat_completions").temperature, true);
});
