// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

type Prune = (providerType: string, modelIds: string[]) => string[];

let vite: ViteDevServer;
let pruneProviderModelIds: Prune;

// /v1/models lists dated ids for pre-4.6 models; undated aliases would be demoted to manual entries.
const DATED = [
  "claude-opus-4-5-20251101",
  "claude-sonnet-4-5-20250929",
  "claude-haiku-4-5-20251001",
  "claude-opus-4-1-20250805",
  "claude-opus-4-20250514",
  "claude-sonnet-4-20250514",
];
const UNDATED = ["claude-opus-5", "claude-opus-4-8", "claude-sonnet-4-6"];

before(async () => {
  vite = await createServer({ appType: "custom", server: { middlewareMode: true } });
  const loaded = await vite.ssrLoadModule("/src/features/chat/sync-external-providers.ts");
  pruneProviderModelIds = loaded.pruneProviderModelIds as Prune;
});

after(async () => {
  await vite.close();
});

test("the frontend prune keeps every dated Anthropic id", () => {
  const live = [...UNDATED, ...DATED];
  assert.deepEqual(pruneProviderModelIds("anthropic", live), live);
});

test("the other providers keep their own prunes", () => {
  assert.deepEqual(pruneProviderModelIds("openai", ["gpt-5.4", "gpt-5.3"]), [
    "gpt-5.4",
  ]);
  assert.deepEqual(
    pruneProviderModelIds("openai_codex", ["gpt-5.4", "gpt-5.3-codex-spark"]),
    ["gpt-5.4"],
  );
  assert.deepEqual(
    pruneProviderModelIds("openrouter", ["openai/gpt-4o", "openai/whisper-1"]),
    ["openai/gpt-4o"],
  );
});
