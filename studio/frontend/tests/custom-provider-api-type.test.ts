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
test("Responses survives provider requests and cache while legacy cache defaults", async () => {
  const api = await vite.ssrLoadModule("/src/features/chat/api/providers-api.ts");
  const providers = await vite.ssrLoadModule("/src/features/chat/external-providers.ts");
  const bodies: Record<string, unknown>[] = [];
  const previous = globalThis.fetch;
  globalThis.fetch = (async (_url, init) => {
    bodies.push(JSON.parse(String(init?.body))); return Response.json({});
  }) as typeof fetch;
  try {
    await api.createProviderConfig({ providerType: "custom", displayName: "Gateway", apiType: "responses" });
    await api.updateProviderConfig("gateway", { apiType: "responses" });
    await api.testProviderConnection({ providerType: "custom", apiKey: "", apiType: "responses" });
  } finally {
    globalThis.fetch = previous;
  }
  assert.equal(bodies.length, 3);
  assert.ok(bodies.every((body) => body.api_type === "responses"));
  const config = { id: "gateway", providerType: "custom", name: "Gateway",
    baseUrl: "https://gateway.example/v1", models: ["responses-only"], createdAt: 1, updatedAt: 1 };
  providers.saveExternalProviders([{ ...config, apiType: "responses" }]);
  assert.equal(providers.loadExternalProviders()[0].apiType, "responses");
  providers.saveExternalProviders([config]);
  assert.equal(providers.loadExternalProviders()[0].apiType, "chat_completions");
});

test("extra header drafts round-trip and reject reserved names", async () => {
  const headers = await vite.ssrLoadModule("/src/features/chat/extra-headers.ts");
  const parsed = headers.parseExtraHeadersDraft(
    "X-LiteLLM-Session-Id: unsloth-{{CHAT_SESSION_ID}}\nX-Title: Unsloth\n",
  );
  assert.equal(parsed["X-LiteLLM-Session-Id"], "unsloth-{{CHAT_SESSION_ID}}");
  assert.equal(
    headers.extraHeadersToDraft(parsed),
    "X-LiteLLM-Session-Id: unsloth-{{CHAT_SESSION_ID}}\nX-Title: Unsloth",
  );
  assert.throws(() => headers.parseExtraHeadersDraft("Authorization: Bearer secret"));
  const json = headers.parseExtraHeadersDraft('{"X-Title":"Studio"}');
  assert.equal(json["X-Title"], "Studio");
  assert.equal(headers.normalizeProviderExtraHeaders({ Authorization: "nope", "X-Title": "ok" })["X-Title"], "ok");
  assert.equal(headers.normalizeProviderExtraHeaders({ Authorization: "nope" }), undefined);
});
