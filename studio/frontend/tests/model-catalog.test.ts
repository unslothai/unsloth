// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const {
  clearProviderModelCatalog,
  providerModelCatalogFetchedAt,
  resolveModelCatalogEntry,
  setProviderModelCatalog,
} = await import("../src/features/chat/model-catalog.ts");
const { getExternalReasoningCapabilities, getPublishedExternalMaxOutputTokens } =
  await import("../src/features/chat/provider-capabilities.ts");
const { providerModelSupportsVision } = await import(
  "../src/features/chat/external-providers.ts"
);

test("an OpenRouter route exposes the snapshot's effort ladder", () => {
  const caps = getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningStyle, "reasoning_effort");
  assert.deepEqual([...caps.reasoningEffortLevels], ["high", "xhigh"]);
  assert.equal(caps.supportsReasoningOff, true);
  assert.equal(caps.reasoningAlwaysOn, false);
});

test("a mandatory-reasoning route keeps the always-on toggle", () => {
  const caps = getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-r1");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningStyle, "enable_thinking");
  assert.equal(caps.reasoningAlwaysOn, true);
  assert.equal(caps.supportsReasoningOff, false);
});

test("a known non-reasoning route hides the control", () => {
  const entry = resolveModelCatalogEntry("openrouter", "openai/gpt-4o");
  assert.ok(entry);
  assert.equal(entry.reasoning, false);
  assert.equal(getExternalReasoningCapabilities("openrouter", "openai/gpt-4o").supportsReasoning, false);
});

test("meta-routers and unknown ids keep the generic OpenRouter toggle", () => {
  for (const model of ["openrouter/free", "openrouter/auto", "acme/never-heard-of-it"]) {
    const caps = getExternalReasoningCapabilities("openrouter", model);
    assert.equal(caps.supportsReasoning, true, model);
    assert.equal(caps.reasoningStyle, "enable_thinking", model);
    assert.equal(caps.supportsReasoningOff, true, model);
    assert.equal(caps.reasoningAlwaysOn, false, model);
  }
});

test("variant suffixes and the ~ prefix resolve through the base id", () => {
  const base = getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro");
  for (const model of ["~deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro:nitro", "DeepSeek/DeepSeek-V4-Pro"]) {
    assert.deepEqual(getExternalReasoningCapabilities("openrouter", model), base, model);
  }
});

test("the live catalog wins over the snapshot and carries the default effort and output cap", () => {
  assert.equal(providerModelCatalogFetchedAt("openrouter"), null);
  setProviderModelCatalog(
    "openrouter",
    [
      {
        id: "deepseek/deepseek-v4-pro",
        input_modalities: ["text"],
        reasoning: { supported_efforts: ["max", "high", "bogus"], mandatory: false, default_effort: "high" },
        max_output_tokens: 65536,
      },
      { id: "acme/plain-chat", input_modalities: ["text", "image"], reasoning: null },
      {
        id: "acme/always-thinking",
        input_modalities: ["text"],
        reasoning: { mandatory: true },
      },
    ],
    1234,
  );
  try {
    assert.equal(providerModelCatalogFetchedAt("openrouter"), 1234);
    const caps = getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro");
    assert.deepEqual([...caps.reasoningEffortLevels], ["high", "max"]);
    assert.equal(caps.defaultEffort, "high");
    assert.equal(getPublishedExternalMaxOutputTokens("openrouter", "deepseek/deepseek-v4-pro"), 65536);

    assert.equal(getExternalReasoningCapabilities("openrouter", "acme/plain-chat").supportsReasoning, false);
    assert.equal(providerModelSupportsVision("openrouter", "acme/plain-chat"), true);

    const mandatory = getExternalReasoningCapabilities("openrouter", "acme/always-thinking");
    assert.equal(mandatory.reasoningStyle, "enable_thinking");
    assert.equal(mandatory.reasoningAlwaysOn, true);
    assert.equal(mandatory.supportsReasoningOff, false);
  } finally {
    clearProviderModelCatalog("openrouter");
  }
  assert.equal(providerModelCatalogFetchedAt("openrouter"), null);
  assert.deepEqual(
    [...getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro").reasoningEffortLevels],
    ["high", "xhigh"],
  );
  assert.equal(getPublishedExternalMaxOutputTokens("openrouter", "deepseek/deepseek-v4-pro"), null);
});

test("vision follows the model rather than the provider type", () => {
  assert.equal(providerModelSupportsVision("openrouter", "deepseek/deepseek-r1"), false);
  assert.equal(providerModelSupportsVision("openrouter", "openai/gpt-5.5"), true);
  assert.equal(providerModelSupportsVision("openrouter", "acme/never-heard-of-it"), true);
  assert.equal(providerModelSupportsVision("deepseek", "deepseek-v4-flash"), true);
  assert.equal(providerModelSupportsVision("deepseek", "deepseek-v4-pro"), false);
  assert.equal(providerModelSupportsVision("deepseek", "deepseek-unknown"), false);
});

test("Ollama models resolve by tag then by base name", () => {
  const exact = getExternalReasoningCapabilities("ollama", "gpt-oss:120b");
  assert.equal(exact.reasoningStyle, "reasoning_effort");
  assert.deepEqual([...exact.reasoningEffortLevels], ["low", "medium", "high"]);
  assert.equal(exact.supportsReasoningOff, false);

  const byBase = getExternalReasoningCapabilities("ollama", "gpt-oss:7b-custom");
  assert.deepEqual(byBase, exact);

  const toggle = getExternalReasoningCapabilities("ollama", "kimi-k2.6:latest");
  assert.equal(toggle.reasoningStyle, "enable_thinking");
  assert.equal(toggle.supportsReasoning, true);
  assert.equal(toggle.supportsReasoningOff, true);

  assert.equal(getExternalReasoningCapabilities("ollama", "mystery:7b").supportsReasoning, false);
});

test("OpenAI ids missing from the prefix tables fall back to the snapshot without an off switch", () => {
  const caps = getExternalReasoningCapabilities("openai", "o4-mini");
  assert.equal(caps.reasoningStyle, "reasoning_effort");
  assert.deepEqual([...caps.reasoningEffortLevels], ["low", "medium", "high"]);
  assert.equal(caps.supportsReasoningOff, false);
  assert.equal(getExternalReasoningCapabilities("openai", "gpt-5.1-chat-latest").supportsReasoning, false);
  assert.equal(getExternalReasoningCapabilities("openai", "gpt-99-unknown").supportsReasoning, false);
});

test("the hand-maintained tables still win for the models they cover", () => {
  const caps = getExternalReasoningCapabilities("anthropic", "claude-opus-5");
  assert.deepEqual([...caps.reasoningEffortLevels], ["none", "low", "medium", "high", "xhigh", "max"]);
  assert.equal(caps.supportsReasoningOff, true);
});
