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

test("explicit-key routing sends extra headers; a saved key leaves them on the record", async () => {
  const routing = await vite.ssrLoadModule("/src/features/chat/utils/chat-title.ts");
  const extraHeaders = { "X-Title": "Unsloth" };
  const provider = {
    id: "gateway",
    providerType: "custom",
    name: "Gateway",
    baseUrl: "https://gateway.example/v1",
    models: ["proxy-model"],
    extraHeaders,
    createdAt: 1,
    updatedAt: 1,
  };
  const saved = await routing.buildExternalRoutingFields({
    provider,
    modelId: "proxy-model",
    apiKey: "",
  });
  assert.equal("extra_headers" in saved, false);
  assert.equal("encrypted_api_key" in saved, false);

  const previous = globalThis.fetch;
  globalThis.fetch = (async () =>
    Response.json({
      public_key: `-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQC1n8QOqkDXkFEOC62kiqZcBCN3
l/DmD+0BGvjg8h1fFJD2Fla1ibcnmKb9Vok+PmR6jm1JX0yu8JHXPw1om01RwQWe
nehl2VzGfdEdNaRoKhW5oVsnnfmxlWJ/qWuV2rDK8DK/6UK9sC/duMkRWaRGdhyl
l+8/fuJc9JDRVzx7HwIDAQAB
-----END PUBLIC KEY-----`,
    })) as typeof fetch;
  try {
    const explicit = await routing.buildExternalRoutingFields({
      provider,
      modelId: "proxy-model",
      apiKey: "sk-browser-held",
    });
    assert.deepEqual(explicit.extra_headers, extraHeaders);
    assert.equal(typeof explicit.encrypted_api_key, "string");
  } finally {
    globalThis.fetch = previous;
  }
});
