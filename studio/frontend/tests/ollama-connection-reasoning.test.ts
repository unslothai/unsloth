// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #9649: the selected model's /api/tags "thinking" capability decides, not the connection.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { getExternalReasoningCapabilities } = await import(
  "../src/features/chat/provider-capabilities.ts"
);

const {
  learnCatalogModelCapabilities,
  providerModelSupportsThinking,
  supportsProviderReasoningToggle,
} = await import("../src/features/chat/external-providers.ts");

learnCatalogModelCapabilities("ollama", [
  {
    id: "thinkingcap-27b-bottlecap:latest",
    capabilities: ["completion", "tools", "thinking"],
  },
  { id: "plainhat-7b:latest", capabilities: ["completion", "tools"] },
  { id: "silent-13b:latest" },
]);

test("an ollama model advertising thinking gets the effort ladder", () => {
  const caps = getExternalReasoningCapabilities(
    "ollama",
    "thinkingcap-27b-bottlecap:latest",
  );
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningStyle, "reasoning_effort");
  assert.equal(caps.supportsReasoningOff, true);
  assert.deepEqual(
    [...caps.reasoningEffortLevels],
    ["low", "medium", "high", "max"],
  );
});

test("a non-thinking model on the same connection keeps no controls", () => {
  const caps = getExternalReasoningCapabilities("ollama", "plainhat-7b:latest");
  assert.equal(caps.supportsReasoning, false);
});

test("a model the catalog never described keeps no controls", () => {
  assert.equal(
    providerModelSupportsThinking("ollama", "silent-13b:latest"),
    null,
  );
  assert.equal(
    getExternalReasoningCapabilities("ollama", "silent-13b:latest")
      .supportsReasoning,
    false,
  );
  assert.equal(
    getExternalReasoningCapabilities("ollama", "never-listed:latest")
      .supportsReasoning,
    false,
  );
});

test("a re-pulled model that stopped thinking loses the ladder", () => {
  learnCatalogModelCapabilities("ollama", [
    { id: "thinkingcap-27b-bottlecap:latest", capabilities: ["completion"] },
  ]);
  assert.equal(
    getExternalReasoningCapabilities(
      "ollama",
      "thinkingcap-27b-bottlecap:latest",
    ).supportsReasoning,
    false,
  );
  learnCatalogModelCapabilities("ollama", [
    {
      id: "thinkingcap-27b-bottlecap:latest",
      capabilities: ["completion", "thinking"],
    },
  ]);
});

test("the connection dialog no longer offers the toggle for ollama", () => {
  assert.equal(supportsProviderReasoningToggle("ollama"), false);
  assert.equal(supportsProviderReasoningToggle("vllm"), true);
  assert.equal(supportsProviderReasoningToggle("openai"), false);
});

test("a flagged vllm connection still gets enable_thinking", () => {
  const caps = getExternalReasoningCapabilities("vllm", "some-local-model", {
    isReasoningProvider: true,
  });
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningStyle, "enable_thinking");
  assert.equal(caps.supportsReasoningOff, true);
});
