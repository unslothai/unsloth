// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The backend has mapped thinking controls onto Ollama's reasoning_effort since
// 1fe27b1b5 (#9649); these pin the frontend half that makes it reachable.
// Ollama answers the question per model — /api/tags names a "thinking"
// capability — so the selected model decides, not a checkbox on the connection.

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

// One connection, both kinds of model — the case a connection-level flag cannot
// answer. Shapes match ProviderModelInfo rows as /api/providers/models sends them.
learnCatalogModelCapabilities("ollama", [
  {
    id: "thinkingcap-27b-bottlecap:latest",
    capabilities: ["completion", "tools", "thinking"],
  },
  { id: "plainhat-7b:latest", capabilities: ["completion", "tools"] },
  // An Ollama too old to report capabilities says nothing about either model.
  { id: "silent-13b:latest" },
]);

test("an ollama model advertising thinking gets the effort ladder", () => {
  const caps = getExternalReasoningCapabilities(
    "ollama",
    "thinkingcap-27b-bottlecap:latest",
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

test("a non-thinking model on the same connection keeps no controls", () => {
  // Ollama errors a thinking request at a model that cannot think, so the
  // sibling model's capability must not leak across the connection.
  const caps = getExternalReasoningCapabilities("ollama", "plainhat-7b:latest");
  assert.equal(caps.supportsReasoning, false);
});

test("a model the catalog never described keeps no controls", () => {
  // Unknown is not yes: a hand-typed id, or a host too old to report.
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
  // The catalog is the truth for rows it describes; a stale yes must not latch.
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
  // Restore the fixture for any later test in this file.
  learnCatalogModelCapabilities("ollama", [
    {
      id: "thinkingcap-27b-bottlecap:latest",
      capabilities: ["completion", "thinking"],
    },
  ]);
});

test("the connection dialog no longer offers the toggle for ollama", () => {
  // Ollama reports thinking per model, so the checkbox would be a worse answer.
  assert.equal(supportsProviderReasoningToggle("ollama"), false);
  // vLLM introduced this contract and has no per-model signal; it keeps it.
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
