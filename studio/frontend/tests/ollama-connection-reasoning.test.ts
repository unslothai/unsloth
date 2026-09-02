// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The backend has mapped thinking controls onto Ollama's reasoning_effort since
// 1fe27b1b5 (#9649); these pin the frontend half that makes it reachable.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { getExternalReasoningCapabilities } = await import(
  "../src/features/chat/provider-capabilities.ts"
);

const { supportsProviderReasoningToggle } = await import(
  "../src/features/chat/external-providers.ts"
);

test("an ollama connection flagged as reasoning gets the effort ladder", () => {
  const caps = getExternalReasoningCapabilities(
    "ollama",
    "thinkingcap-27b-bottlecap:latest",
    { isReasoningProvider: true },
  );
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningStyle, "reasoning_effort");
  // #9649 is a model thinking by default with no way to turn it off.
  assert.equal(caps.supportsReasoningOff, true);
  // The non-"none" half of ollama's accepted set; "none" is what off sends.
  assert.deepEqual(
    [...caps.reasoningEffortLevels],
    ["low", "medium", "high", "max"],
  );
});

test("an unflagged ollama connection keeps no reasoning controls", () => {
  // Ollama errors a thinking request at a model that cannot think.
  const caps = getExternalReasoningCapabilities(
    "ollama",
    "thinkingcap-27b-bottlecap:latest",
    {},
  );
  assert.equal(caps.supportsReasoning, false);
});

test("the connection dialog offers the reasoning toggle for ollama", () => {
  assert.equal(supportsProviderReasoningToggle("ollama"), true);
  // vLLM introduced this contract; it must survive ollama joining it.
  assert.equal(supportsProviderReasoningToggle("vllm"), true);
  assert.equal(supportsProviderReasoningToggle("openai"), false);
});
