// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// the per-connection local-tools opt-in: which provider types may offer it,
// when it counts as on, and that a backend sync never resurrects or loses it.

import assert from "node:assert/strict";
import test from "node:test";

import type { ExternalProviderConfig } from "../src/features/chat/external-providers.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { providerLocalToolsEnabled, supportsProviderLocalTools } = await import(
  "../src/features/chat/external-providers.ts"
);
const { mergeLocalProviderOptions } = await import(
  "../src/features/chat/sync-external-providers.ts"
);

function provider(
  providerType: string,
  overrides: Partial<ExternalProviderConfig> = {},
): ExternalProviderConfig {
  return {
    id: "p1",
    providerType,
    name: "My server",
    baseUrl: "http://localhost:8000/v1",
    models: ["qwen3-14b"],
    createdAt: 1,
    updatedAt: 2,
    ...overrides,
  };
}

test("only self-hosted OpenAI-compat connections offer the local-tools opt-in", () => {
  // mirrors local_tool_loop_provider_types in
  // studio/backend/core/inference/external_tool_loop.py.
  for (const providerType of ["vllm", "ollama", "llama_cpp", "custom"]) {
    assert.equal(supportsProviderLocalTools(providerType), true, providerType);
  }
  for (const providerType of [
    "openai",
    "anthropic",
    "gemini",
    "openrouter",
    "kimi",
    "mistral",
  ]) {
    assert.equal(supportsProviderLocalTools(providerType), false, providerType);
  }
  assert.equal(supportsProviderLocalTools(null), false);
  assert.equal(supportsProviderLocalTools(undefined), false);
});

test("the opt-in defaults off and never applies to a hosted provider", () => {
  assert.equal(providerLocalToolsEnabled(provider("vllm")), false);
  assert.equal(
    providerLocalToolsEnabled(provider("vllm", { enableLocalTools: false })),
    false,
  );
  assert.equal(
    providerLocalToolsEnabled(provider("vllm", { enableLocalTools: true })),
    true,
  );
  // a stale flag on a hosted connection must not light the pills: those
  // providers run their own server-side tools.
  assert.equal(
    providerLocalToolsEnabled(provider("openai", { enableLocalTools: true })),
    false,
  );
  assert.equal(providerLocalToolsEnabled(null), false);
  assert.equal(providerLocalToolsEnabled(undefined), false);
});

test("a backend sync keeps the browser-local opt-in", () => {
  // the backend never stores this flag, so the rebuilt config carries no value
  // and the local one has to survive the merge.
  const merged = mergeLocalProviderOptions(
    provider("vllm", { enableLocalTools: true }),
    provider("vllm"),
  );
  assert.equal(merged.enableLocalTools, true);
});

test("a sync of a hosted connection clears the opt-in", () => {
  const merged = mergeLocalProviderOptions(
    provider("openai", { enableLocalTools: true }),
    provider("openai"),
  );
  assert.equal(merged.enableLocalTools, undefined);
});

test("an unknown connection syncs to the off default", () => {
  const merged = mergeLocalProviderOptions(undefined, provider("ollama"));
  assert.equal(merged.enableLocalTools, undefined);
  assert.equal(providerLocalToolsEnabled(merged), false);
});
