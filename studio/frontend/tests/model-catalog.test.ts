// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const {
  clearProviderModelCatalog,
  modelCatalogVersion,
  providerModelCatalogFetchedAt,
  resolveModelCatalogEntry,
  setProviderModelCatalog,
  subscribeModelCatalog,
} = await import("../src/features/chat/model-catalog.ts");
const {
  getExternalReasoningCapabilities,
  getPublishedExternalMaxOutputTokens,
  reasoningFieldsAfterCatalogRefresh,
} = await import("../src/features/chat/provider-capabilities.ts");
const { providerModelSupportsVision } = await import(
  "../src/features/chat/external-providers.ts"
);

const LIVE = [
  {
    id: "deepseek/deepseek-v4-pro",
    input_modalities: ["text"],
    reasoning: { supported_efforts: ["max", "high", "bogus"], mandatory: false, default_effort: "high" },
    max_output_tokens: 65536,
  },
  { id: "acme/plain-chat", input_modalities: ["text", "image"], reasoning: null },
  { id: "acme/always-thinking", input_modalities: ["text"], reasoning: { mandatory: true } },
];

function withLiveCatalog(run: () => void): void {
  setProviderModelCatalog("openrouter", LIVE, 1234);
  try {
    run();
  } finally {
    clearProviderModelCatalog("openrouter");
  }
}

test("without a live catalog every OpenRouter route keeps the generic toggle", () => {
  assert.equal(providerModelCatalogFetchedAt("openrouter"), null);
  for (const model of ["openrouter/free", "openrouter/auto", "acme/never-heard-of-it", "deepseek/deepseek-v4-pro"]) {
    const caps = getExternalReasoningCapabilities("openrouter", model);
    assert.equal(caps.supportsReasoning, true, model);
    assert.equal(caps.reasoningStyle, "enable_thinking", model);
    assert.equal(caps.supportsReasoningOff, true, model);
    assert.equal(caps.reasoningAlwaysOn, false, model);
  }
  assert.equal(getPublishedExternalMaxOutputTokens("openrouter", "deepseek/deepseek-v4-pro"), null);
});

test("the live catalog carries the effort ladder, default effort, output cap and mandatory routes", () => {
  withLiveCatalog(() => {
    assert.equal(providerModelCatalogFetchedAt("openrouter"), 1234);
    const caps = getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro");
    assert.equal(caps.reasoningStyle, "reasoning_effort");
    assert.deepEqual([...caps.reasoningEffortLevels], ["none", "high", "max"]);
    assert.equal(caps.defaultEffort, "high");
    assert.equal(getPublishedExternalMaxOutputTokens("openrouter", "deepseek/deepseek-v4-pro"), 65536);

    assert.equal(getExternalReasoningCapabilities("openrouter", "acme/plain-chat").supportsReasoning, false);

    const mandatory = getExternalReasoningCapabilities("openrouter", "acme/always-thinking");
    assert.equal(mandatory.reasoningStyle, "enable_thinking");
    assert.equal(mandatory.reasoningAlwaysOn, true);
    assert.equal(mandatory.supportsReasoningOff, false);
    assert.equal(
      getExternalReasoningCapabilities("openrouter", "openrouter/free").reasoningStyle,
      "enable_thinking",
      "meta-routers keep the generic toggle",
    );
  });
  assert.equal(providerModelCatalogFetchedAt("openrouter"), null);
  assert.equal(
    getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro").reasoningStyle,
    "enable_thinking",
  );
});

test("variant suffixes and the ~ prefix resolve through the base id", () => {
  withLiveCatalog(() => {
    const base = getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro");
    for (const model of ["~deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro:nitro", "DeepSeek/DeepSeek-V4-Pro"]) {
      assert.deepEqual(getExternalReasoningCapabilities("openrouter", model), base, model);
    }
    assert.ok(resolveModelCatalogEntry("openrouter", "deepseek/deepseek-v4-pro:nitro"));
  });
});

test("OpenRouter vision follows the live catalog's input modalities", () => {
  withLiveCatalog(() => {
    assert.equal(providerModelSupportsVision("openrouter", "acme/plain-chat"), true);
    assert.equal(providerModelSupportsVision("openrouter", "deepseek/deepseek-v4-pro"), false);
    assert.equal(providerModelSupportsVision("openrouter", "acme/never-heard-of-it"), true);
  });
  assert.equal(providerModelSupportsVision("openrouter", "deepseek/deepseek-v4-pro"), true);
});

test("providers other than OpenRouter keep their existing resolvers", () => {
  withLiveCatalog(() => {
    const opus = getExternalReasoningCapabilities("anthropic", "claude-opus-5");
    assert.deepEqual([...opus.reasoningEffortLevels], ["none", "low", "medium", "high", "xhigh", "max"]);
    for (const [provider, model] of [
      ["deepseek", "deepseek-v4-pro"],
      ["ollama", "gpt-oss:120b"],
      ["vllm", "openai/gpt-oss-20b"],
      ["llama_cpp", "Qwen3-0.6B-Q4_K_M.gguf"],
      ["anthropic", "claude-3-7-sonnet-20250219"],
    ] as const) {
      assert.equal(getExternalReasoningCapabilities(provider, model).supportsReasoning, false, `${provider} ${model}`);
    }
    assert.equal(providerModelSupportsVision("deepseek", "deepseek-v4-pro"), false);
  });
});

test("a live catalog default of none survives when the model can switch reasoning off", () => {
  setProviderModelCatalog(
    "openrouter",
    [
      {
        id: "openai/gpt-5.1",
        reasoning: { supported_efforts: ["high", "medium", "low", "none"], mandatory: false, default_effort: "none" },
      },
      {
        id: "acme/always-on",
        reasoning: { supported_efforts: ["high", "none"], mandatory: true, default_effort: "none" },
      },
    ],
    1,
  );
  try {
    const gpt51 = getExternalReasoningCapabilities("openrouter", "openai/gpt-5.1");
    assert.deepEqual([...gpt51.reasoningEffortLevels], ["none", "low", "medium", "high"]);
    assert.equal(gpt51.defaultEffort, "none");
    const alwaysOn = getExternalReasoningCapabilities("openrouter", "acme/always-on");
    assert.equal(alwaysOn.supportsReasoningOff, false);
    assert.equal(alwaysOn.defaultEffort, null, "a mandatory model cannot default to off");
  } finally {
    clearProviderModelCatalog("openrouter");
  }
});

test("every catalog write notifies subscribers, so a composer already on screen re-renders", () => {
  let notified = 0;
  const unsubscribe = subscribeModelCatalog(() => {
    notified += 1;
  });
  const before = modelCatalogVersion();
  try {
    setProviderModelCatalog("openrouter", [{ id: "acme/late", reasoning: { supported_efforts: ["high"] } }], 1);
    assert.equal(notified, 1);
    clearProviderModelCatalog("openrouter");
    assert.equal(notified, 2);
    clearProviderModelCatalog("openrouter");
    assert.equal(notified, 2, "clearing an absent catalog changes nothing");
  } finally {
    unsubscribe();
  }
  assert.notEqual(modelCatalogVersion(), before);
  setProviderModelCatalog("openrouter", [{ id: "acme/late" }], 1);
  clearProviderModelCatalog("openrouter");
  assert.equal(notified, 2, "an unsubscribed listener is not called");
});

test("a catalog that lands after selection refreshes the stored reasoning fields without resetting a valid effort", () => {
  const generic = getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro");
  assert.equal(generic.reasoningStyle, "enable_thinking");
  withLiveCatalog(() => {
    const caps = getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro");
    const kept = reasoningFieldsAfterCatalogRefresh({ reasoningEffort: "high", reasoningEnabled: false }, caps);
    assert.equal(kept.supportsReasoning, true);
    assert.equal(kept.reasoningStyle, "reasoning_effort");
    assert.deepEqual([...kept.reasoningEffortLevels], ["none", "high", "max"]);
    assert.equal(kept.reasoningEffort, "high", "a chosen effort the ladder still offers is kept");
    assert.equal(kept.reasoningEnabled, false, "a toggleable model keeps the stored choice");

    const clamped = reasoningFieldsAfterCatalogRefresh({ reasoningEffort: "medium", reasoningEnabled: true }, caps);
    assert.equal(clamped.reasoningEffort, "high", "an effort the ladder dropped is clamped onto an offered one");

    const alwaysOn = reasoningFieldsAfterCatalogRefresh(
      { reasoningEffort: "medium", reasoningEnabled: false },
      getExternalReasoningCapabilities("openrouter", "acme/always-thinking"),
    );
    assert.equal(alwaysOn.reasoningAlwaysOn, true);
    assert.equal(alwaysOn.reasoningEnabled, true, "a model without an off switch is forced on");
  });
});
