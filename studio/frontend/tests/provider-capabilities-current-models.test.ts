// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  getExternalMaxOutputTokens,
  getExternalReasoningCapabilities,
  providerSupportsBuiltinCodeExecution,

  providerSupportsBuiltinWebSearch,
  providerSupportsFastMode,
} = await import("../src/features/chat/provider-capabilities.ts");

const { providerModelSupportsVision, setProviderModelCapabilities } = await import(
  "../src/features/chat/external-providers.ts"
);

// The picker's default_models list (backend core/inference/providers.py) grew a
// Claude 5 / gpt-5.6 / gemini-3.6 generation. Every table here is prefix-based,
// so an un-widened prefix silently drops the control instead of failing loudly:
// a model with no reasoning entry loses its Thinking picker entirely.

test("Claude 5 and Opus 4.8 expose the adaptive effort ladder", () => {
  for (const model of [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
  ]) {
    const caps = getExternalReasoningCapabilities("anthropic", model);
    assert.equal(caps.supportsReasoning, true, model);
    assert.equal(caps.supportsReasoningOff, true, model);
    assert.deepEqual(
      [...caps.reasoningEffortLevels],
      ["none", "low", "medium", "high", "xhigh", "max"],
      model,
    );
  }
});

test("Fable 5 thinks always, so no off switch is offered", () => {
  // `thinking.type: "disabled"` 400s on Fable/Mythos 5.
  const caps = getExternalReasoningCapabilities("anthropic", "claude-fable-5");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.supportsReasoningOff, false);
  assert.ok(![...caps.reasoningEffortLevels].includes("none"));
});

test("fast mode is offered on Opus 5 / 4.8 and nowhere else", () => {
  for (const model of ["claude-opus-5", "claude-opus-4-8-2026-02-01"]) {
    assert.equal(providerSupportsFastMode("anthropic", model), true, model);
  }
  // 4.7 errors on `speed`; 4.6 accepts it but answers at standard speed.
  for (const model of ["claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-5"]) {
    assert.equal(providerSupportsFastMode("anthropic", model), false, model);
  }
});

test("the gpt-5.6 family gets the gpt-5.5 reasoning ladder", () => {
  for (const model of ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]) {
    const caps = getExternalReasoningCapabilities("openai", model);
    assert.equal(caps.supportsReasoning, true, model);
    assert.equal(caps.supportsReasoningOff, true, model);
    // The API rejects "minimal" on this family.
    assert.deepEqual(
      [...caps.reasoningEffortLevels],
      ["none", "low", "medium", "high", "xhigh"],
      model,
    );
    assert.equal(getExternalMaxOutputTokens("openai", model), 128000, model);
  }
});

test("ChatGPT subscription models expose Studio-owned search and code tools", () => {

  setProviderModelCapabilities("openai_codex", {
    "gpt-5.3-codex-spark": { vision: false, studio_tools: true },
    "gpt-5.4": { vision: true, studio_tools: true },
    "gpt-5.6-sol": { vision: true, studio_tools: true },
  });
  for (const model of ["gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.6-sol"]) {
    const caps = getExternalReasoningCapabilities("openai_codex", model);
    assert.equal(caps.supportsReasoning, true, model);
    assert.equal(caps.reasoningStyle, "reasoning_effort", model);
    assert.equal(getExternalMaxOutputTokens("openai_codex", model), 128000, model);
    assert.equal(providerSupportsBuiltinWebSearch("openai_codex", model), true, model);
    assert.equal(providerSupportsBuiltinCodeExecution("openai_codex", model), true, model);
  }
});


test("ChatGPT subscription vision gating follows the curated model", () => {

  setProviderModelCapabilities("openai_codex", {
    "gpt-5.3-codex-spark": { vision: false, studio_tools: true },
    "gpt-5.6-sol": { vision: true, studio_tools: true },
  });
  assert.equal(
    providerModelSupportsVision("openai_codex", "gpt-5.3-codex-spark"),
    false,
  );
  assert.equal(providerModelSupportsVision("openai_codex", "gpt-5.6-sol"), true);
});


test("Gemini 3.x minors keep the thinkingLevel ladder", () => {
  // gemini-3.6-flash must not fall through to the 2.5 integer-budget branch.
  for (const model of ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview"]) {
    const caps = getExternalReasoningCapabilities("gemini", model);
    assert.equal(caps.reasoningStyle, "reasoning_effort", model);
    assert.deepEqual(
      [...caps.reasoningEffortLevels],
      ["minimal", "low", "medium", "high"],
      model,
    );
  }
  const pro = getExternalReasoningCapabilities("gemini", "gemini-3.1-pro-preview");
  assert.deepEqual([...pro.reasoningEffortLevels], ["low", "medium", "high"]);
});

test("new Anthropic and OpenAI ids keep their max-output cap and code pill", () => {
  for (const model of ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8"]) {
    assert.equal(getExternalMaxOutputTokens("anthropic", model), 128000, model);
    assert.equal(providerSupportsBuiltinCodeExecution("anthropic", model), true, model);
  }
  assert.equal(
    providerSupportsBuiltinCodeExecution("openai", "gpt-5.6-sol", "https://api.openai.com/v1"),
    true,
  );
});

test("generic Custom connections use only their explicit max-output override", () => {
  // Model names that resemble known hosted families must not change Custom's cap.
  assert.equal(getExternalMaxOutputTokens("custom", "gpt-5.6-sol"), 32768);
  assert.equal(getExternalMaxOutputTokens("custom", "claude-opus-5"), 32768);

  assert.equal(
    getExternalMaxOutputTokens("custom", "any/provider-model", 131072),
    131072,
  );
  assert.equal(getExternalMaxOutputTokens("custom", null, 65536), 65536);

  // Invalid persisted values fail closed to the conservative default.
  assert.equal(getExternalMaxOutputTokens("custom", "model", 63), 32768);
  assert.equal(getExternalMaxOutputTokens("custom", "model", 65536.5), 32768);
  assert.equal(
    getExternalMaxOutputTokens("custom", "model", Number.MAX_SAFE_INTEGER + 1),
    32768,
  );

  // The override is provider-owned, so values above Studio's context-length
  // convention remain valid as long as they round-trip safely through JSON.
  assert.equal(getExternalMaxOutputTokens("custom", "model", 1048577), 1048577);
  assert.equal(
    getExternalMaxOutputTokens("custom", "model", Number.MAX_SAFE_INTEGER),
    Number.MAX_SAFE_INTEGER,
  );
});

test("Custom overrides cannot alter documented caps for other providers", () => {
  assert.equal(getExternalMaxOutputTokens("openai", "gpt-5.6-sol", 64), 128000);
  assert.equal(getExternalMaxOutputTokens("anthropic", "claude-opus-5", 64), 128000);
  assert.equal(getExternalMaxOutputTokens("vllm", "gpt-5.6-sol", 131072), 32768);
});
