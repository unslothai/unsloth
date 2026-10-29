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

test("System One connections stay decision-only after a reload", async () => {
  const api = await vite.ssrLoadModule("/src/features/chat/api/providers-api.ts");
  const providers = await vite.ssrLoadModule("/src/features/chat/external-providers.ts");
  const bodies: Record<string, unknown>[] = [];
  const previous = globalThis.fetch;
  globalThis.fetch = (async (_url, init) => {
    bodies.push(JSON.parse(String(init?.body))); return Response.json({});
  }) as typeof fetch;
  try {
    await api.createProviderConfig({ providerType: "custom", displayName: "Decider", apiType: "systemone" });
  } finally {
    globalThis.fetch = previous;
  }
  assert.equal(bodies[0].api_type, "systemone");

  const decider = { id: "decider", providerType: "custom", name: "Decider", baseUrl: "http://localhost:8080/v1",
    models: ["jev-latest"], createdAt: 1, updatedAt: 1, ...providers.connectionApiFields("systemone") };
  providers.saveExternalProviders([decider]);
  assert.ok(providers.isDecisionConnection(providers.loadExternalProviders()[0]));
  for (const providerType of ["typesafe", "liquid"]) {
    assert.ok(providers.isDecisionConnection({ providerType }), providerType);
  }
  for (const apiType of ["chat_completions", "responses", undefined]) {
    const chat = { providerType: "custom", ...providers.connectionApiFields(apiType) };
    assert.equal(providers.isDecisionConnection(chat), false, String(apiType));
  }
  assert.equal(providers.isDecisionConnection({ providerType: "openrouter" }), false);
});
