// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Qwen3 thinking table must not replace sampling a server's unsloth.ini set, and only that.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { layersQwenThinkingDefaults, mergeBackendRecommendedInference } =
  await import("../src/features/chat/presets/preset-policy.ts");
const { resolveQwenThinkingParams } = await import(
  "../src/features/chat/utils/qwen-sampling-table.ts"
);
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);

const QWEN = "unsloth/Qwen3-0.6B-GGUF";

/** What performLoad and the status merge do, in order: the response's defaults, then the table. */
function afterLoad(
  response: {
    model_ini_sampling?: boolean;
    inference: Record<string, number>;
  },
  current = DEFAULT_INFERENCE_PARAMS,
) {
  let params = mergeBackendRecommendedInference({
    current: { ...current, checkpoint: QWEN },
    response: { is_gguf: true, ...response },
    modelId: QWEN,
    presetSource: "builtin-default",
    loadedContextLength: 4096,
  });
  const qwen = resolveQwenThinkingParams(QWEN, true);
  if (
    qwen &&
    layersQwenThinkingDefaults("builtin-default", response.model_ini_sampling)
  ) {
    params = { ...params, ...qwen };
  }
  return params;
}

test("a load run with the unsloth.ini keeps the INI's sampling on the Default preset", () => {
  const params = afterLoad({
    model_ini_sampling: true,
    inference: { temperature: 0.42, top_p: 0.77, top_k: 17, min_p: 0.03 },
  });
  assert.equal(params.temperature, 0.42);
  assert.equal(params.topP, 0.77);
  assert.equal(params.topK, 17);
  assert.equal(params.minP, 0.03);
});

test("the same model without the file goes back to its normal recommendation", () => {
  const params = afterLoad({
    model_ini_sampling: false,
    inference: { temperature: 0.42, top_p: 0.77, top_k: 17 },
  });
  assert.equal(params.temperature, 0.6);
  assert.equal(params.topP, 0.95);
  assert.equal(params.topK, 20);
  // A server that predates the field is a load without the file.
  assert.equal(layersQwenThinkingDefaults("builtin-default", undefined), true);
  assert.equal(layersQwenThinkingDefaults("custom", false), false);
});

test("both sites that lay the thinking table over a load ask the helper", () => {
  assert.match(
    readSrc("features/chat/hooks/use-chat-model-runtime.ts"),
    /layersQwenThinkingDefaults\(\s*store\.activePresetSource,\s*loadResponse\.model_ini_sampling,\s*\)/,
  );
  assert.match(
    readSrc("features/chat/lib/apply-inference-status-to-store.ts"),
    /layersQwenThinkingDefaults\(current\.activePresetSource, status\.model_ini_sampling\)/,
  );
});

test("an INI with only performance settings keeps the Qwen3 thinking defaults", () => {
  // The backend reports model_ini_sampling false for it, and inference holds only model defaults.
  const params = afterLoad({
    model_ini_sampling: false,
    inference: { temperature: 0.7, top_p: 0.8, top_k: 20 },
  });
  assert.equal(params.temperature, 0.6);
  assert.equal(params.topP, 0.95);
});

test("an INI repeat-penalty reaches the slider; a load without one leaves it alone", () => {
  const withIni = afterLoad({
    model_ini_sampling: true,
    inference: { temperature: 0.42, repetition_penalty: 1.1 },
  });
  assert.equal(withIni.repetitionPenalty, 1.1);
  const kept = afterLoad(
    { inference: { temperature: 0.7 } },
    { ...DEFAULT_INFERENCE_PARAMS, repetitionPenalty: 1.25 },
  );
  assert.equal(kept.repetitionPenalty, 1.25);
});
