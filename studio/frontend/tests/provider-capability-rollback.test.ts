// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  PROVIDER_CAPABILITY_WILDCARD,
  providerModelSupportsStudioTools,
  pruneProviderModelCapabilities,
  setProviderModelCapabilities,
} = await import("../src/features/chat/external-providers.ts");

// The localStorage capability map outlives the backend, so the sync must prune unlisted providers.

test("a provider the registry stopped listing loses its capability", () => {
  setProviderModelCapabilities("llama_cpp", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
  });
  setProviderModelCapabilities("openai", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
  });
  assert.equal(providerModelSupportsStudioTools("llama_cpp", "any-gguf"), true);

  pruneProviderModelCapabilities(["openai"]);

  assert.equal(providerModelSupportsStudioTools("llama_cpp", "any-gguf"), null);
  assert.equal(providerModelSupportsStudioTools("openai", "gpt-5.4"), true);
});

test("an empty registry clears everything rather than freezing it", () => {
  setProviderModelCapabilities("vllm", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
  });

  // Clearing is the safe direction: an unknown capability reads as not capable.
  pruneProviderModelCapabilities([]);

  assert.equal(providerModelSupportsStudioTools("vllm", "m"), null);
});

test("a row that stops declaring the capability is corrected in place", () => {
  setProviderModelCapabilities("ollama", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
  });

  setProviderModelCapabilities("ollama", {});
  pruneProviderModelCapabilities(["ollama"]);

  assert.equal(providerModelSupportsStudioTools("ollama", "llama4"), null);
});

test("pruning a registry that lists everything changes nothing", () => {
  setProviderModelCapabilities("gemini", {
    "gemini-3-pro": { studio_tools: true, vision: true },
  });

  pruneProviderModelCapabilities(["gemini", "openai", "anthropic"]);

  assert.equal(providerModelSupportsStudioTools("gemini", "gemini-3-pro"), true);
});
