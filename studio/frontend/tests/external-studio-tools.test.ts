// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { buildExternalEnabledTools } from "../src/features/chat/api/external-tool-payload.ts";
import {
  type ExternalProviderConfig,
  normalizeProvider,
} from "../src/features/chat/external-providers.ts";

function provider(studioToolExecution?: boolean): ExternalProviderConfig {
  return {
    id: "provider-1",
    providerType: "ollama",
    name: "Ollama",
    baseUrl: "http://localhost:11434/v1",
    models: ["qwen3"],
    studioToolExecution: studioToolExecution as boolean,
    createdAt: 1,
    updatedAt: 1,
  };
}

test("Studio tool execution defaults off for legacy provider records", () => {
  const legacy = provider();
  (legacy as Partial<ExternalProviderConfig>).studioToolExecution = undefined;

  assert.equal(normalizeProvider(legacy).studioToolExecution, false);
});

test("Studio tool execution opt-in is independent of provider type", () => {
  for (const providerType of [
    "openai",
    "anthropic",
    "gemini",
    "custom",
    "ollama",
  ]) {
    const configured = provider(true);
    configured.providerType = providerType;
    assert.equal(normalizeProvider(configured).studioToolExecution, true);
  }
});

test("Studio execution preserves unrelated provider-hosted tools", () => {
  assert.deepEqual(
    buildExternalEnabledTools({
      studioToolExecution: true,
      webSearch: true,
      webFetch: true,
      codeExecution: true,
      imageGeneration: true,
    }),
    ["web_search", "web_fetch", "python", "terminal", "image_generation"],
  );
});

test("disabled Studio execution keeps provider-native Search and Code", () => {
  assert.deepEqual(
    buildExternalEnabledTools({
      studioToolExecution: false,
      webSearch: true,
      webFetch: true,
      codeExecution: true,
      imageGeneration: true,
    }),
    ["web_search", "web_fetch", "code_execution", "image_generation"],
  );
});
