// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { getExternalReasoningCapabilities } = await import(
  "../src/features/chat/provider-capabilities.ts"
);

const {
  getProviderModelReasoningCapabilities,
  setProviderModelReasoningCapabilities,
} = await import("../src/features/chat/external-providers.ts");

// Reasoning capabilities for connected llama.cpp models are probed per model on
// an explicit "Detect reasoning" action and cached. The cached result must win
// over the hardcoded provider tables, so a Qwen3.8 served by llama.cpp surfaces
// the same effort ladder the local GGUF path derives.

test("backend-probed enable_thinking_effort ladder wins for llama.cpp", () => {
  setProviderModelReasoningCapabilities("llama_cpp", "Qwen3.8-14B", {
    supports_reasoning: true,
    reasoning_style: "enable_thinking_effort",
    reasoning_effort_levels: ["low", "medium", "xhigh"],
    reasoning_always_on: false,
  });

  // The getter matches case-insensitively.
  const caps = getExternalReasoningCapabilities("llama_cpp", "qwen3.8-14b");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningStyle, "enable_thinking_effort");
  assert.equal(caps.supportsReasoningOff, true);
  assert.deepEqual([...caps.reasoningEffortLevels], ["low", "medium", "xhigh"]);
});

test("probed reasoning_effort style has no off switch", () => {
  setProviderModelReasoningCapabilities("llama_cpp", "gpt-oss-20b", {
    supports_reasoning: true,
    reasoning_style: "reasoning_effort",
    reasoning_effort_levels: ["low", "medium", "high"],
    reasoning_always_on: false,
  });

  const caps = getExternalReasoningCapabilities("llama_cpp", "gpt-oss-20b");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.supportsReasoningOff, false);
  assert.deepEqual([...caps.reasoningEffortLevels], ["low", "medium", "high"]);
});

test("a probed non-reasoning model is cached as unsupported", () => {
  setProviderModelReasoningCapabilities("llama_cpp", "plain-model", {
    supports_reasoning: false,
    reasoning_style: "enable_thinking",
    reasoning_effort_levels: [],
    reasoning_always_on: false,
  });

  assert.ok(
    getProviderModelReasoningCapabilities("llama_cpp", "plain-model"),
  );
  const caps = getExternalReasoningCapabilities("llama_cpp", "plain-model");
  assert.equal(caps.supportsReasoning, false);
});

test("clearing a cached entry falls back to no reasoning", () => {
  setProviderModelReasoningCapabilities("llama_cpp", "qwen3.8-14b", null);

  assert.equal(
    getProviderModelReasoningCapabilities("llama_cpp", "qwen3.8-14b"),
    undefined,
  );
  const caps = getExternalReasoningCapabilities("llama_cpp", "qwen3.8-14b");
  assert.equal(caps.supportsReasoning, false);
});

test("hosted provider ladders are not shadowed by the probe map", () => {
  const caps = getExternalReasoningCapabilities("anthropic", "claude-opus-5");
  assert.equal(caps.supportsReasoning, true);
  assert.deepEqual(
    [...caps.reasoningEffortLevels],
    ["none", "low", "medium", "high", "xhigh", "max"],
  );
});
