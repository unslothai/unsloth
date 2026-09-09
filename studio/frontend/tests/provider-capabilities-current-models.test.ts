// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  clampReasoningEffortToLevels,
  getExternalMaxOutputTokens,
  getExternalReasoningCapabilities,
  providerSupportsBuiltinCodeExecution,

  providerSupportsBuiltinWebSearch,
  providerSupportsFastMode,
} = await import("../src/features/chat/provider-capabilities.ts");

const { providerModelSupportsVision, setProviderModelCapabilities } = await import(
  "../src/features/chat/external-providers.ts"
);

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

test("Astra exposes mandatory reasoning with its full effort ladder", () => {
  const caps = getExternalReasoningCapabilities("openai_codex", "gpt-6-astra");
  assert.equal(caps.supportsReasoning, true);
  assert.equal(caps.reasoningStyle, "reasoning_effort");
  assert.equal(caps.supportsReasoningOff, false);
  assert.deepEqual(
    [...caps.reasoningEffortLevels],
    ["low", "medium", "high", "xhigh", "max"],
  );
  for (const effort of ["none", "minimal"] as const) {
    assert.equal(clampReasoningEffortToLevels(effort, caps.reasoningEffortLevels), "low");
  }
  assert.equal(clampReasoningEffortToLevels("max", caps.reasoningEffortLevels), "max");
});

// A local GGUF's ladder is whatever its template branches on, so it can skip scale rungs.
test("a narrower local ladder clamps to the nearest rung, not to the weakest", () => {
  const qwen38 = ["low", "medium", "xhigh"] as const;

  assert.equal(clampReasoningEffortToLevels("high", qwen38), "medium");
  assert.equal(clampReasoningEffortToLevels("max", qwen38), "xhigh");
  for (const effort of qwen38) {
    assert.equal(clampReasoningEffortToLevels(effort, qwen38), effort);
  }
  // No lower neighbour, so the weakest rung is right.
  assert.equal(clampReasoningEffortToLevels("none", qwen38), "low");
  assert.equal(clampReasoningEffortToLevels("minimal", qwen38), "low");
});

test("the xhigh -> max alias still wins over the neighbour search", () => {
  // Claude 4.6 renamed the rung rather than dropping it: same level, not a clamp.
  assert.equal(
    clampReasoningEffortToLevels("xhigh", ["low", "medium", "high", "max"]),
    "max",
  );
});

test("clamping down never lands on thinking-off", () => {
  for (const effort of ["minimal", "low", "medium"] as const) {
    assert.equal(clampReasoningEffortToLevels(effort, ["none", "high"]), "high", effort);
  }
  assert.equal(
    clampReasoningEffortToLevels("minimal", ["none", "low", "medium", "high"]),
    "low",
  );
  assert.equal(clampReasoningEffortToLevels("none", ["none", "high"]), "none");
  assert.equal(clampReasoningEffortToLevels("high", ["none"]), "none");
});

test("the ladder order a caller passes does not change the clamp", () => {
  // A local ladder comes from a chat-template scan, so nothing keeps it ascending.
  const ascending = ["low", "medium", "xhigh"] as const;
  const descending = ["xhigh", "medium", "low"] as const;
  for (const effort of ["none", "minimal", "low", "medium", "high", "xhigh", "max"] as const) {
    assert.equal(
      clampReasoningEffortToLevels(effort, descending),
      clampReasoningEffortToLevels(effort, ascending),
      effort,
    );
  }
});

test("every shipped effort ladder is ordered weakest first", () => {
  // The Think menu renders table order, and `fallbackExternalEffort` in chat-adapter reads
  // `reasoningEffortLevels[0]` as the weakest rung; out-of-order tables break both.
  const models: Array<[string, string]> = [
    ["anthropic", "claude-opus-5"],
    ["anthropic", "claude-opus-4-6"],
    ["anthropic", "claude-opus-4-5"],
    ["anthropic", "claude-fable-5"],
    ["openai", "gpt-5.6-sol"],
    ["openai", "gpt-5.5-pro"],
    ["openai", "gpt-5.1-codex-max"],
    ["openai", "gpt-5.1"],
    ["openai", "gpt-5"],
    ["openai", "o3"],
    ["openai_codex", "gpt-6-astra"],
    ["gemini", "gemini-2.5-flash-lite"],
    ["gemini", "gemini-2.5-pro"],
    ["gemini", "gemini-2.5-flash"],
    ["gemini", "gemini-3-pro"],
    ["gemini", "gemini-3-flash"],
    ["mistral", "magistral-medium-latest"],
    ["mistral", "mistral-small-latest"],
    ["openrouter", "openai/gpt-5.5"],
  ];
  const scale = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
  for (const [provider, model] of models) {
    const levels = [
      ...getExternalReasoningCapabilities(provider, model).reasoningEffortLevels,
    ];
    const ranks = levels.map((level) => scale.indexOf(level));
    assert.ok(!ranks.includes(-1), `${model} lists a level off the scale: ${levels}`);
    assert.deepEqual(
      ranks,
      [...ranks].sort((a, b) => a - b),
      `${model} ladder is not weakest first: ${levels}`,
    );
    assert.equal(new Set(levels).size, levels.length, `${model} repeats a level`);
  }
});

test("the effort scale matches the backend's _REASONING_EFFORT_SCALE", () => {
  // The clamp ranks against the frontend copy while detect_reasoning_flags builds local
  // ladders from the backend one, so drift hands the clamp a level it cannot rank.
  const here = path.dirname(fileURLToPath(import.meta.url));
  const frontend = readFileSync(
    path.join(here, "../src/features/chat/provider-capabilities.ts"),
    "utf8",
  );
  const backend = readFileSync(
    path.join(here, "../../backend/core/inference/llama_cpp.py"),
    "utf8",
  );
  const frontendScale = frontend
    .match(/const REASONING_EFFORT_SCALE = \[([\s\S]*?)\] as const/)?.[1]
    .match(/"([a-z]+)"/g)
    ?.map((quoted) => quoted.slice(1, -1));
  const backendScale = backend
    .match(/^_REASONING_EFFORT_SCALE = \(([^)]*)\)/m)?.[1]
    .match(/"([a-z]+)"/g)
    ?.map((quoted) => quoted.slice(1, -1));
  assert.ok(frontendScale, "frontend REASONING_EFFORT_SCALE not found");
  assert.ok(backendScale, "backend _REASONING_EFFORT_SCALE not found");
  assert.deepEqual(frontendScale, backendScale);
});

test("Astra reasoning does not enable unrelated model families", () => {
  for (const model of ["gpt-6-other", "gpt-60-astra"]) {
    assert.equal(getExternalReasoningCapabilities("openai_codex", model).supportsReasoning, false);
  }
});

test("the gpt-5.1 and gpt-5.2 ladders drop minimal for none", () => {
  // "minimal" was replaced by "none" from 5.1 on, and offering it fails the
  // turn with "does not support 'minimal' with this model".
  const ladders: Array<[string, readonly string[]]> = [
    ["gpt-5.2", ["none", "low", "medium", "high", "xhigh"]],
    ["gpt-5.1", ["none", "low", "medium", "high"]],
  ];
  for (const [model, levels] of ladders) {
    const caps = getExternalReasoningCapabilities("openai", model);
    assert.equal(caps.supportsReasoningOff, true, model);
    assert.deepEqual([...caps.reasoningEffortLevels], levels, model);
  }
  // The Codex tunings keep reasoning mandatory: no minimal, and no none on
  // the 5.1 line. Only codex-max has xhigh, so it sorts first.
  const codexLadders: Array<[string, readonly string[]]> = [
    ["gpt-5-codex", ["low", "medium", "high"]],
    ["gpt-5.1-codex", ["low", "medium", "high"]],
    ["gpt-5.1-codex-mini", ["low", "medium", "high"]],
    ["gpt-5.1-codex-max", ["low", "medium", "high", "xhigh"]],
  ];
  for (const [model, levels] of codexLadders) {
    const caps = getExternalReasoningCapabilities("openai", model);
    assert.equal(caps.supportsReasoningOff, false, model);
    assert.deepEqual([...caps.reasoningEffortLevels], levels, model);
  }
  // Bare gpt-5 keeps the old ladder, so the splits must not swallow it.
  const five = getExternalReasoningCapabilities("openai", "gpt-5");
  assert.equal(five.supportsReasoningOff, false);
  assert.deepEqual(
    [...five.reasoningEffortLevels],
    ["minimal", "low", "medium", "high"],
  );
});

test("the chat-latest aliases advertise no reasoning at all", () => {
  // They are non-reasoning, and the family prefixes would otherwise swallow
  // them: `gpt-5.1-chat-latest` starts with `gpt-5.1`. Advertising reasoning
  // makes the adapter send `reasoning_effort` on every turn, which the
  // Responses API rejects with "Unsupported parameter: 'reasoning.effort' is
  // not supported with this model" -- so the model never answers at all.
  for (const model of [
    "gpt-5-chat-latest",
    "gpt-5.1-chat-latest",
    "gpt-5.2-chat-latest",
    "gpt-5.3-chat-latest",
    // Azure names its deployment without the `-latest` tail.
    "gpt-5-chat",
  ]) {
    const caps = getExternalReasoningCapabilities("openai", model);
    assert.equal(caps.supportsReasoning, false, model);
  }
  // The reasoning families themselves must keep theirs.
  for (const model of ["gpt-5.1", "gpt-5.2", "gpt-5", "gpt-5.3-codex"]) {
    assert.equal(
      getExternalReasoningCapabilities("openai", model).supportsReasoning,
      true,
      model,
    );
  }
  // `chatgpt-4o-latest` is a different shape and was already non-reasoning.
  assert.equal(
    getExternalReasoningCapabilities("openai", "chatgpt-4o-latest")
      .supportsReasoning,
    false,
  );
});

test("ChatGPT subscription models expose Unsloth-owned search and code tools", () => {

  setProviderModelCapabilities("openai_codex", {
    "gpt-5.3-codex-spark": { vision: false, studio_tools: true },
    "gpt-5.4": { vision: true, studio_tools: true },
    "gpt-5.6-sol": { vision: true, studio_tools: true },
    "gpt-6-astra": { vision: true, studio_tools: true },
  });
  for (const model of ["gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.6-sol", "gpt-6-astra"]) {
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
