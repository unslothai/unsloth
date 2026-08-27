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

const {
  providerModelSupportsVision,
  setProviderModelCapabilities,
  reasoningFieldsForProviderSave,
  supportsPerModelReasoningPin,
  supportsProviderReasoningToggle,
} = await import("../src/features/chat/external-providers.ts");

// Every capability table is prefix-based, so an un-widened prefix silently drops a
// control instead of failing loudly: a model with no reasoning entry loses its
// Thinking picker entirely.

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
  // `thinking.type: "disabled"` 400s on Fable/Mythos 5
  const caps = getExternalReasoningCapabilities("anthropic", "claude-fable-5");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.supportsReasoningOff, false);
  assert.ok(![...caps.reasoningEffortLevels].includes("none"));
});

test("fast mode is offered on Opus 5 / 4.8 and nowhere else", () => {
  for (const model of ["claude-opus-5", "claude-opus-4-8-2026-02-01"]) {
    assert.equal(providerSupportsFastMode("anthropic", model), true, model);
  }
  // 4.7 errors on `speed`; 4.6 accepts it but answers at standard speed
  for (const model of ["claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-5"]) {
    assert.equal(providerSupportsFastMode("anthropic", model), false, model);
  }
});

test("the gpt-5.6 family gets the gpt-5.5 reasoning ladder", () => {
  for (const model of ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]) {
    const caps = getExternalReasoningCapabilities("openai", model);
    assert.equal(caps.supportsReasoning, true, model);
    assert.equal(caps.supportsReasoningOff, true, model);
    // the API rejects "minimal" on this family
    assert.deepEqual(
      [...caps.reasoningEffortLevels],
      ["none", "low", "medium", "high", "xhigh"],
      model,
    );
    assert.equal(getExternalMaxOutputTokens("openai", model), 128000, model);
  }
});

test("ChatGPT subscription models expose Unsloth-owned search and code tools", () => {

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
  // gemini-3.6-flash must not fall through to the 2.5 integer-budget branch
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
  // no capability row targets `custom`, so a model id resembling a hosted family
  // never enters the decision
  assert.equal(getExternalMaxOutputTokens("custom", "gpt-5.6-sol"), 32768);
  assert.equal(getExternalMaxOutputTokens("custom", "claude-opus-5"), 32768);

  assert.equal(
    getExternalMaxOutputTokens("custom", "any/provider-model", 131072),
    131072,
  );
  assert.equal(getExternalMaxOutputTokens("custom", null, 65536), 65536);

  // invalid persisted values fail closed to the conservative default
  assert.equal(getExternalMaxOutputTokens("custom", "model", 63), 32768);
  assert.equal(getExternalMaxOutputTokens("custom", "model", 65536.5), 32768);
  assert.equal(
    getExternalMaxOutputTokens("custom", "model", Number.MAX_SAFE_INTEGER + 1),
    32768,
  );

  // the override is provider-owned, so values above Unsloth's context-length convention
  // stay valid as long as they round-trip safely through JSON
  assert.equal(getExternalMaxOutputTokens("custom", "model", 1048577), 1048577);
  assert.equal(
    getExternalMaxOutputTokens("custom", "model", Number.MAX_SAFE_INTEGER),
    Number.MAX_SAFE_INTEGER,
  );
});

test("a connection override cannot raise a documented per-model cap", () => {
  assert.equal(getExternalMaxOutputTokens("openai", "gpt-5.6-sol", 999999), 128000);
  assert.equal(getExternalMaxOutputTokens("anthropic", "claude-opus-5", 999999), 128000);
  // it lowers one, though: a gateway or spend policy below the published cap is real
  assert.equal(getExternalMaxOutputTokens("openai", "gpt-5.6-sol", 8192), 8192);
  // a vLLM server hosting an id borrowed from OpenAI has no documented cap of its own
  assert.equal(getExternalMaxOutputTokens("vllm", "gpt-5.6-sol", 131072), 131072);
});

test("Ollama offers the connection-level reasoning toggle, like vLLM", () => {
  assert.equal(supportsProviderReasoningToggle("ollama"), true);
  assert.equal(supportsProviderReasoningToggle("vllm"), true);
  assert.equal(supportsProviderReasoningToggle("llama_cpp"), false);
  assert.equal(supportsProviderReasoningToggle("custom"), false);
  assert.equal(supportsPerModelReasoningPin("ollama"), true);
  assert.equal(supportsPerModelReasoningPin("vllm"), false);
});

test("Ollama hides Thinking unless the connection is flagged as a reasoning model", () => {
  const model = "thinkingcap-27b-bottlecap:latest";
  const unmarked = getExternalReasoningCapabilities("ollama", model);
  assert.equal(unmarked.supportsReasoning, false);
  assert.equal(
    getExternalReasoningCapabilities("ollama", model, {
      isReasoningProvider: false,
    }).supportsReasoning,
    false,
  );

  const flagged = getExternalReasoningCapabilities("ollama", model, {
    isReasoningProvider: true,
    reasoningModelIds: [model],
  });
  assert.equal(flagged.supportsReasoning, true);
  assert.equal(flagged.reasoningStyle, "reasoning_effort");
  assert.equal(flagged.supportsReasoningOff, true);
  assert.deepEqual([...flagged.reasoningEffortLevels], [
    "none",
    "low",
    "medium",
    "high",
  ]);
});

test("Ollama only advertises Thinking for models pinned on the connection", () => {
  const thinking = "thinkingcap-27b-bottlecap:latest";
  const instruct = "llama3.2:latest";
  const mixed = {
    isReasoningProvider: true,
    reasoningModelIds: [thinking],
  } as const;

  const pinned = getExternalReasoningCapabilities("ollama", thinking, mixed);
  assert.equal(pinned.supportsReasoning, true);
  assert.equal(pinned.reasoningStyle, "reasoning_effort");

  const other = getExternalReasoningCapabilities("ollama", instruct, mixed);
  assert.equal(other.supportsReasoning, false);

  const noneMarked = getExternalReasoningCapabilities("ollama", thinking, {
    isReasoningProvider: true,
    reasoningModelIds: [],
  });
  assert.equal(noneMarked.supportsReasoning, false);

  // Legacy connection-wide pin: no per-model list yet.
  const legacy = getExternalReasoningCapabilities("ollama", instruct, {
    isReasoningProvider: true,
  });
  assert.equal(legacy.supportsReasoning, true);

  // vLLM stays connection-wide even if a leftover Ollama pin list is present.
  assert.equal(
    getExternalReasoningCapabilities("vllm", instruct, {
      isReasoningProvider: true,
      reasoningModelIds: [],
    }).supportsReasoning,
    true,
  );
});

test("Ollama GPT-OSS reasoning stays on and only offers supported levels", () => {
  for (const model of ["gpt-oss:20b", "registry.ollama.ai/library/gpt-oss:120b"]) {
    const caps = getExternalReasoningCapabilities("ollama", model, {
      isReasoningProvider: true,
      reasoningModelIds: [model],
    });
    assert.equal(caps.supportsReasoning, true, model);
    assert.equal(caps.reasoningAlwaysOn, true, model);
    assert.equal(caps.supportsReasoningOff, false, model);
    assert.deepEqual([...caps.reasoningEffortLevels], ["low", "medium", "high"], model);
  }
});

test("Ollama save persists an explicit per-model pin, including none", () => {
  assert.deepEqual(
    reasoningFieldsForProviderSave(
      "ollama",
      true,
      ["thinkingcap-27b-bottlecap:latest", "llama3.2:latest"],
      ["thinkingcap-27b-bottlecap:latest", "dropped"],
    ),
    {
      isReasoningModel: true,
      reasoningModelIds: ["thinkingcap-27b-bottlecap:latest"],
    },
  );
  assert.deepEqual(
    reasoningFieldsForProviderSave(
      "ollama",
      true,
      ["llama3.2:latest"],
      [],
    ),
    { isReasoningModel: true, reasoningModelIds: [] },
  );
  assert.deepEqual(reasoningFieldsForProviderSave("vllm", true, ["m"], ["m"]), {
    isReasoningModel: true,
    reasoningModelIds: undefined,
  });
});
