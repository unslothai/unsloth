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
const {
  getExternalReasoningCapabilities,
  getPublishedExternalMaxOutputTokens,
  reasoningFieldsAfterCatalogRefresh,
} = await import("../src/features/chat/provider-capabilities.ts");
const { providerModelSupportsVision } = await import(
  "../src/features/chat/external-providers.ts"
);

test("an OpenRouter route exposes the snapshot's effort ladder", () => {
  const caps = getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningStyle, "reasoning_effort");
  assert.deepEqual([...caps.reasoningEffortLevels], ["none", "high", "xhigh"]);
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
    assert.deepEqual([...caps.reasoningEffortLevels], ["none", "high", "max"]);
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
    ["none", "high", "xhigh"],
  );
  assert.equal(getPublishedExternalMaxOutputTokens("openrouter", "deepseek/deepseek-v4-pro"), null);
});

test("vision follows the model rather than the provider type", () => {
  assert.equal(providerModelSupportsVision("openrouter", "deepseek/deepseek-r1"), false);
  assert.equal(providerModelSupportsVision("openrouter", "openai/gpt-5.5"), true);
  assert.equal(providerModelSupportsVision("openrouter", "acme/never-heard-of-it"), true);
  // The catalog lists image input for v4-flash, but the backend strips DeepSeek images before sending.
  assert.equal(providerModelSupportsVision("deepseek", "deepseek-v4-flash"), false);
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

  const qwen3 = getExternalReasoningCapabilities("ollama", "qwen3:0.6b");
  assert.equal(qwen3.reasoningStyle, "enable_thinking");
  assert.equal(qwen3.supportsReasoning, true);
  assert.equal(getExternalReasoningCapabilities("ollama", "mystery:7b").supportsReasoning, false);
});

test("an Ollama instruct-only tag never inherits thinking from a sibling or another provider", () => {
  for (const id of ["qwen3-vl:8b-instruct", "qwen3:4b-instruct-2507"]) {
    assert.equal(getExternalReasoningCapabilities("ollama", id).supportsReasoning, false, id);
  }
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

test("DeepSeek direct maps the snapshot ladder onto its low / high / max wire", () => {
  const caps = getExternalReasoningCapabilities("deepseek", "deepseek-v4-flash");
  assert.equal(caps.reasoningStyle, "reasoning_effort");
  assert.deepEqual([...caps.reasoningEffortLevels], ["none", "low", "high", "max"]);
  assert.equal(caps.supportsReasoningOff, true);
  assert.equal(getExternalReasoningCapabilities("deepseek", "deepseek-unknown").supportsReasoning, false);
});

test("Qwen and Kimi only have a wire toggle, so a ladder collapses to on/off", () => {
  for (const [provider, model] of [["qwen", "qwen3.5-plus"], ["kimi", "kimi-k3"]] as const) {
    const caps = getExternalReasoningCapabilities(provider, model);
    assert.equal(caps.reasoningStyle, "enable_thinking", model);
    assert.equal(caps.supportsReasoning, true, model);
    assert.equal(caps.supportsReasoningOff, true, model);
  }
  assert.equal(getExternalReasoningCapabilities("kimi", "kimi-k2.6").supportsReasoning, true);
});

test("Mistral models outside the table get the documented none / high form", () => {
  const caps = getExternalReasoningCapabilities("mistral", "mistral-medium-latest");
  assert.equal(caps.reasoningStyle, "reasoning_effort");
  assert.deepEqual([...caps.reasoningEffortLevels], ["none", "high"]);
  const known = getExternalReasoningCapabilities("mistral", "mistral-small-latest");
  assert.deepEqual([...known.reasoningEffortLevels], ["none", "high"]);
  const magistral = getExternalReasoningCapabilities("mistral", "magistral-small");
  assert.equal(magistral.reasoningStyle, "enable_thinking");
  assert.equal(magistral.reasoningAlwaysOn, true);
  assert.equal(magistral.supportsReasoningOff, false);
  assert.equal(getExternalReasoningCapabilities("mistral", "mistral-large-latest").supportsReasoning, false);
});

test("Hugging Face router forwards the snapshot ladder verbatim", () => {
  const caps = getExternalReasoningCapabilities("huggingface", "openai/gpt-oss-120b");
  assert.deepEqual([...caps.reasoningEffortLevels], ["low", "medium", "high"]);
  assert.equal(caps.supportsReasoningOff, false);
});

test("vLLM and llama.cpp resolve by bare model name and clamp to low / medium / high", () => {
  const vllm = getExternalReasoningCapabilities("vllm", "openai/gpt-oss-20b");
  assert.deepEqual([...vllm.reasoningEffortLevels], ["low", "medium", "high"]);
  const gguf = getExternalReasoningCapabilities("llama_cpp", "gpt-oss-20b-Q4_K_M.gguf");
  assert.deepEqual(gguf, vllm);
  const qwen = getExternalReasoningCapabilities("vllm", "Qwen/Qwen3-14B");
  assert.equal(qwen.reasoningStyle, "enable_thinking");
  assert.equal(qwen.supportsReasoningOff, true);
  const family = getExternalReasoningCapabilities("llama_cpp", "Qwen3-0.6B-Q4_K_M.gguf");
  assert.equal(family.reasoningStyle, "enable_thinking");
  assert.equal(family.supportsReasoning, true);
  assert.equal(getExternalReasoningCapabilities("llama_cpp", "Llama-3.2-1B-Instruct-Q4_K_M.gguf").supportsReasoning, false);
  assert.equal(getExternalReasoningCapabilities("llama_cpp", "mystery-7b.gguf").supportsReasoning, false);
  assert.equal(getExternalReasoningCapabilities("custom", "openai/gpt-oss-20b").supportsReasoning, false);
});

test("Gemini aliases outside the prefix tables fall back to the snapshot", () => {
  const caps = getExternalReasoningCapabilities("gemini", "gemini-omni-flash-preview");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningStyle, "enable_thinking");
  assert.equal(getExternalReasoningCapabilities("gemini", "gemini-3.1-pro-preview").reasoningStyle, "reasoning_effort");
});

test("a served models.dev catalog outranks the bundled snapshot and feeds the name index", async () => {
  const { modelsDevCatalogFetchedAt, setModelsDevCatalog } = await import(
    "../src/features/chat/model-catalog.ts"
  );
  assert.equal(modelsDevCatalogFetchedAt(), null);
  assert.equal(getExternalReasoningCapabilities("deepseek", "deepseek-v9-ultra").supportsReasoning, false);
  setModelsDevCatalog({
    fetched_at: 1_700_000_000,
    providers: {
      deepseek: { "deepseek-v9-ultra": { reasoning: true, toggle: true, efforts: ["low", "max"], input: ["text"] } },
      openrouter: { "deepseek/deepseek-v4-pro": { reasoning: true, toggle: true, efforts: ["low"], input: ["text", "image"] } },
      huggingface: { "acme/newmodel-70b": { reasoning: true, efforts: ["low", "high"], input: ["text"] } },
    },
  });
  try {
    assert.equal(modelsDevCatalogFetchedAt(), 1_700_000_000);
    const fresh = getExternalReasoningCapabilities("deepseek", "deepseek-v9-ultra");
    assert.deepEqual([...fresh.reasoningEffortLevels], ["none", "low", "max"]);
    assert.deepEqual(
      [...getExternalReasoningCapabilities("openrouter", "deepseek/deepseek-v4-pro").reasoningEffortLevels],
      ["none", "low"],
    );
    assert.equal(providerModelSupportsVision("openrouter", "deepseek/deepseek-v4-pro"), true);
    const local = getExternalReasoningCapabilities("llama_cpp", "newmodel-8b-Q4_K_M.gguf");
    assert.deepEqual([...local.reasoningEffortLevels], ["low", "high"]);
    assert.equal(getExternalReasoningCapabilities("ollama", "gpt-oss:120b").supportsReasoning, true);
  } finally {
    setModelsDevCatalog({ fetched_at: 0, providers: {} });
  }
  assert.equal(getExternalReasoningCapabilities("deepseek", "deepseek-v9-ultra").supportsReasoning, false);
});

test("a served bucket overrides its own models without hiding the rest of the bundled one", async () => {
  // The browser cache survives an app upgrade and the backend serves an expired disk copy
  // while offline, so a bucket written by an older release can omit models the newer
  // bundled snapshot knows. Replacing the namespace wholesale dropped their controls.
  // DeepSeek, not Ollama: the Ollama branch has a by-name fallback that hides the defect.
  const { setModelsDevCatalog } = await import("../src/features/chat/model-catalog.ts");
  const bundled = getExternalReasoningCapabilities("deepseek", "deepseek-v4-pro");
  assert.deepEqual([...bundled.reasoningEffortLevels], ["none", "high", "max"]);
  setModelsDevCatalog({
    fetched_at: 1_700_000_000,
    providers: { deepseek: { "deepseek-v5": { reasoning: true, efforts: ["low"], toggle: true, input: ["text"] } } },
  });
  try {
    const served = getExternalReasoningCapabilities("deepseek", "deepseek-v5");
    assert.deepEqual([...served.reasoningEffortLevels], ["none", "low"]);
    const survivor = getExternalReasoningCapabilities("deepseek", "deepseek-v4-pro");
    assert.deepEqual([...survivor.reasoningEffortLevels], ["none", "high", "max"]);
  } finally {
    setModelsDevCatalog({ fetched_at: 0, providers: {} });
  }
});

test("every catalog write notifies subscribers, so a composer already on screen re-renders", async () => {
  const { modelCatalogVersion, setModelsDevCatalog, subscribeModelCatalog } = await import(
    "../src/features/chat/model-catalog.ts"
  );
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
    setModelsDevCatalog({ fetched_at: 0, providers: {} });
    assert.equal(notified, 3);
  } finally {
    unsubscribe();
  }
  assert.notEqual(modelCatalogVersion(), before);
  setProviderModelCatalog("openrouter", [{ id: "acme/late" }], 1);
  clearProviderModelCatalog("openrouter");
  assert.equal(notified, 3, "an unsubscribed listener is not called");
});

test("Gemini image models keep no reasoning control even though the catalog lists one", () => {
  for (const model of ["gemini-3-pro-image", "gemini-2.5-flash-image"]) {
    assert.equal(resolveModelCatalogEntry("gemini", model)?.reasoning, true, model);
    assert.equal(getExternalReasoningCapabilities("gemini", model).supportsReasoning, false, model);
  }
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

test("a catalog that lands after selection refreshes the stored reasoning fields without resetting a valid effort", () => {
  setProviderModelCatalog(
    "openrouter",
    [
      { id: "acme/late-ladder", reasoning: { supported_efforts: ["none", "high", "max"], mandatory: false } },
      { id: "acme/always-thinking", reasoning: { mandatory: true } },
    ],
    1,
  );
  try {
    const caps = getExternalReasoningCapabilities("openrouter", "acme/late-ladder");
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
  } finally {
    clearProviderModelCatalog("openrouter");
  }
});

test("a mandatory OpenRouter alias keeps its catalog effort ladder", () => {
  setProviderModelCatalog("openrouter", [{
    id: "~google/gemini-pro-latest",
    reasoning: {
      mandatory: true,
      supported_efforts: ["high", "medium", "low"],
      default_effort: "medium",
    },
  }]);
  try {
    const caps = getExternalReasoningCapabilities("openrouter", "~google/gemini-pro-latest");
    assert.equal(caps.reasoningStyle, "reasoning_effort");
    assert.deepEqual([...caps.reasoningEffortLevels], ["low", "medium", "high"]);
    assert.equal(caps.supportsReasoningOff, false);
    assert.equal(caps.defaultEffort, "medium");
  } finally {
    clearProviderModelCatalog("openrouter");
  }
  const fallback = getExternalReasoningCapabilities("openrouter", "~google/gemini-pro-latest");
  assert.equal(fallback.reasoningAlwaysOn, true);
  assert.equal(fallback.supportsReasoningOff, false);
});
