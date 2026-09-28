// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer, type ViteDevServer } from "vite";

type Capability = { vision?: boolean; studio_tools?: boolean };
type Merge = (
  stored: Record<string, Capability> | undefined,
  registryCapabilities: Record<string, Capability> | undefined,
  supportsStudioTools: boolean | undefined,
) => Record<string, Capability>;

let vite: ViteDevServer;
let mergeLearnedModelCapabilities: Merge;

const REGISTRY: Record<string, Capability> = {
  "gpt-5.4": { vision: true, studio_tools: true },
};

before(async () => {
  vite = await createServer({ appType: "custom", server: { middlewareMode: true } });
  const loaded = await vite.ssrLoadModule("/src/features/chat/sync-external-providers.ts");
  mergeLearnedModelCapabilities = loaded.mergeLearnedModelCapabilities as Merge;
});

after(async () => {
  await vite.close();
});

test("a plan-listed slug's capability survives a registry rewrite", () => {
  // The startup credential bootstrap syncs before anything fetches /codex/models, so
  // dropping the learned entry here lets the composer offer attachments again.
  const merged = mergeLearnedModelCapabilities(
    { "gpt-5.7-nova": { vision: false }, "gpt-5.4": { vision: true, studio_tools: true } },
    REGISTRY,
    true,
  );
  assert.deepEqual(merged["gpt-5.7-nova"], { vision: false });
});

test("the registry wins for models it describes", () => {
  const merged = mergeLearnedModelCapabilities(
    { "gpt-5.4": { vision: false } },
    REGISTRY,
    true,
  );
  assert.deepEqual(merged["gpt-5.4"], { vision: true, studio_tools: true });
});

test("the studio-tools wildcard is re-established, never inherited stale", () => {
  const merged = mergeLearnedModelCapabilities({ "*": { studio_tools: true } }, REGISTRY, false);
  assert.deepEqual(merged["*"], { studio_tools: false });
});

test("no stored capabilities is the plain registry map", () => {
  assert.deepEqual(mergeLearnedModelCapabilities(undefined, REGISTRY, undefined), REGISTRY);
});
