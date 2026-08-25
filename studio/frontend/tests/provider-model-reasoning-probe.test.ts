// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { getExternalReasoningCapabilities } = await import(
  "../src/features/chat/provider-capabilities.ts"
);

const { setProviderModelReasoningCapabilities } = await import(
  "../src/features/chat/external-providers.ts"
);

// Reasoning capabilities for self-hosted connections are probed from the server's
// Jinja chat template by the backend and must win over the hardcoded provider
// tables, so a Qwen3.8 served by llama.cpp surfaces the same effort ladder the
// local GGUF path derives.

test("backend-probed enable_thinking_effort ladder wins for llama.cpp", () => {
  setProviderModelReasoningCapabilities("llama_cpp", [
    {
      // Server-reported case; the getter matches case-insensitively.
      id: "Qwen3.8-14B",
      display_name: "Qwen3.8-14B",
      reasoning: {
        supports_reasoning: true,
        reasoning_style: "enable_thinking_effort",
        reasoning_effort_levels: ["low", "medium", "xhigh"],
        reasoning_always_on: false,
      },
    },
  ]);

  const caps = getExternalReasoningCapabilities("llama_cpp", "qwen3.8-14b");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningStyle, "enable_thinking_effort");
  assert.equal(caps.supportsReasoningOff, true);
  assert.deepEqual([...caps.reasoningEffortLevels], ["low", "medium", "xhigh"]);
});

test("probed reasoning_effort style has no off switch", () => {
  setProviderModelReasoningCapabilities("llama_cpp", [
    {
      id: "gpt-oss-20b",
      display_name: "gpt-oss-20b",
      reasoning: {
        supports_reasoning: true,
        reasoning_style: "reasoning_effort",
        reasoning_effort_levels: ["low", "medium", "high"],
        reasoning_always_on: false,
      },
    },
  ]);

  const caps = getExternalReasoningCapabilities("llama_cpp", "gpt-oss-20b");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.supportsReasoningOff, false);
  assert.deepEqual([...caps.reasoningEffortLevels], ["low", "medium", "high"]);
});

test("an unprobed self-hosted model keeps today's no-reasoning fallback", () => {
  setProviderModelReasoningCapabilities("ollama", []);

  const caps = getExternalReasoningCapabilities("ollama", "plain:latest");
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
