// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

type Resolve = (
  providerType: string,
  serverModels: string[],
  savedModels: string[],
  defaultModels: string[],
) => string[];

let vite: ViteDevServer;
let resolveSyncedModelIds: Resolve;

const RETIRED = "gpt-5.3-codex-spark";
const SEED = ["gpt-5.4", "gpt-5.5"];

before(async () => {
  vite = await createServer({ appType: "custom", server: { middlewareMode: true } });
  const loaded = await vite.ssrLoadModule("/src/features/chat/sync-external-providers.ts");
  resolveSyncedModelIds = loaded.resolveSyncedModelIds as Resolve;
});

after(async () => {
  await vite.close();
});

test("a saved selection of only the retired slug falls back to the seed", () => {
  // This is the upgrade path the PR exists for: the connection would otherwise sync to
  // an empty model list, vanish from the picker, and backfill [] that the backend 400s.
  const resolved = resolveSyncedModelIds("openai_codex", [], [RETIRED], SEED);
  assert.deepEqual(resolved, SEED);
});

test("a saved selection that survives pruning is kept", () => {
  const resolved = resolveSyncedModelIds("openai_codex", [], [RETIRED, "gpt-5.5"], SEED);
  assert.deepEqual(resolved, ["gpt-5.5"]);
});

test("the server list wins whenever it has anything", () => {
  const resolved = resolveSyncedModelIds("openai_codex", ["gpt-5.4"], [RETIRED], SEED);
  assert.deepEqual(resolved, ["gpt-5.4"]);
});

test("an unrelated provider type is untouched by codex pruning", () => {
  const resolved = resolveSyncedModelIds("openai", [], [RETIRED], SEED);
  assert.deepEqual(resolved, [RETIRED]);
});

test("nothing anywhere still yields the seed", () => {
  assert.deepEqual(resolveSyncedModelIds("openai_codex", [], [], SEED), SEED);
});
