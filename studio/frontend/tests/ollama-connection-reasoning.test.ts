// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The backend has mapped thinking controls onto Ollama's reasoning_effort field
// since 1fe27b1b5 (#9649), but the frontend never advertised the capability, so
// the Thinking control stayed hidden and the mapper was unreachable from the
// UI. Ollama's OpenAI-compat endpoint does not advertise reasoning per model —
// the same shape vLLM has — so the same connection-level "Reasoning model"
// toggle is the contract, pinned here end to end: the dialog shows the toggle,
// and a flagged connection resolves to an effort ladder the backend accepts.

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
  // Off must stay reachable: the whole report is a model burning 89s of
  // default thinking on a trivial prompt with no way to turn it off.
  assert.equal(caps.supportsReasoningOff, true);
  // Exactly the non-"none" values ollama's /v1/chat/completions accepts; the
  // backend's _OLLAMA_REASONING_EFFORTS passes each through untranslated.
  assert.deepEqual(
    [...caps.reasoningEffortLevels],
    ["low", "medium", "high", "max"],
  );
});

test("an unflagged ollama connection keeps no reasoning controls", () => {
  // Ollama errors a thinking request at a model that cannot think, so the
  // control only appears when the user says the connection serves one.
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
