// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  PROVIDER_CAPABILITY_WILDCARD,
  providerModelSupportsStudioTools,
  setProviderModelCapabilities,
} = await import("../src/features/chat/external-providers.ts");

// Self-hosted models have no per-model registry entry, so the wildcard capability must apply.

test("a provider-level capability applies to any model on that provider", () => {
  setProviderModelCapabilities("llama_cpp", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
  });

  for (const modelId of ["Qwen3.5-4B-MTP", "some/local-gguf", "whatever-the-user-typed"]) {
    assert.equal(providerModelSupportsStudioTools("llama_cpp", modelId), true);
  }
});

test("a per-model entry wins over the provider-level default", () => {
  setProviderModelCapabilities("vllm", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
    "text-only-model": { studio_tools: false },
  });

  assert.equal(providerModelSupportsStudioTools("vllm", "text-only-model"), false);
  assert.equal(providerModelSupportsStudioTools("vllm", "anything-else"), true);
});

test("a provider that declares nothing stays unknown", () => {
  setProviderModelCapabilities("anthropic", {});

  assert.equal(providerModelSupportsStudioTools("anthropic", "claude-opus-5"), null);
  assert.equal(providerModelSupportsStudioTools("never-registered", "m"), null);
});

test("a missing provider type is unknown rather than capable", () => {
  assert.equal(providerModelSupportsStudioTools(null, "m"), null);
  assert.equal(providerModelSupportsStudioTools(undefined, "m"), null);
});

test("a known provider answers even before a model is chosen", () => {
  setProviderModelCapabilities("ollama", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
  });

  // The composer asks before a model is picked; null would grey out the pills.
  assert.equal(providerModelSupportsStudioTools("ollama", null), true);
});
