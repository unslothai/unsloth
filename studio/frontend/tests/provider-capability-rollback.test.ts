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

// The capability map is localStorage, so it outlives the backend that wrote it:
// a browser carries it across a downgrade, and the sync only ever writes the
// rows the registry returned. A provider that has been hidden, or that a rolled
// back backend has never heard of, is simply absent from the response -- so
// nothing corrects it, and its last-known studio_tools: true keeps the composer
// offering a loop that backend cannot run. The sync has to converge, not latch.

test("a provider the registry stopped listing loses its capability", () => {
  setProviderModelCapabilities("llama_cpp", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
  });
  setProviderModelCapabilities("openai", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
  });
  assert.equal(providerModelSupportsStudioTools("llama_cpp", "any-gguf"), true);

  // The rolled-back backend still knows openai, but not llama_cpp.
  pruneProviderModelCapabilities(["openai"]);

  assert.equal(providerModelSupportsStudioTools("llama_cpp", "any-gguf"), null);
  assert.equal(providerModelSupportsStudioTools("openai", "gpt-5.4"), true);
});

test("an empty registry clears everything rather than freezing it", () => {
  setProviderModelCapabilities("vllm", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
  });

  // "No provider types exist" is a real answer, and clearing is the safe
  // direction: an unknown capability reads as null, which every caller treats
  // as not capable.
  pruneProviderModelCapabilities([]);

  assert.equal(providerModelSupportsStudioTools("vllm", "m"), null);
});

test("a row that stops declaring the capability is corrected in place", () => {
  setProviderModelCapabilities("ollama", {
    [PROVIDER_CAPABILITY_WILDCARD]: { studio_tools: true },
  });

  // An older backend returns the row without supports_studio_tools, so the sync
  // rebuilds the entry with no wildcard. The write replaces rather than merges,
  // which is what makes this half already converge.
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
