// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Qwen3 thinking table must not replace the sampling a server launched with its unsloth.ini reports.

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
function afterLoad(response: {
  model_ini_applied?: boolean;
  inference: Record<string, number>;
}) {
  let params = mergeBackendRecommendedInference({
    current: { ...DEFAULT_INFERENCE_PARAMS, checkpoint: QWEN },
    response: { is_gguf: true, ...response },
    modelId: QWEN,
    presetSource: "builtin-default",
    loadedContextLength: 4096,
  });
  const qwen = resolveQwenThinkingParams(QWEN, true);
  if (
    qwen &&
    layersQwenThinkingDefaults("builtin-default", response.model_ini_applied)
  ) {
    params = { ...params, ...qwen };
  }
  return params;
}

test("a load run with the unsloth.ini keeps the INI's sampling on the Default preset", () => {
  const params = afterLoad({
    model_ini_applied: true,
    inference: { temperature: 0.42, top_p: 0.77, top_k: 17, min_p: 0.03 },
  });
  assert.equal(params.temperature, 0.42);
  assert.equal(params.topP, 0.77);
  assert.equal(params.topK, 17);
  assert.equal(params.minP, 0.03);
});

test("the same model without the file goes back to its normal recommendation", () => {
  const params = afterLoad({
    model_ini_applied: false,
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
    /layersQwenThinkingDefaults\(\s*store\.activePresetSource,\s*loadResponse\.model_ini_applied,\s*\)/,
  );
  assert.match(
    readSrc("features/chat/lib/apply-inference-status-to-store.ts"),
    /layersQwenThinkingDefaults\(current\.activePresetSource, status\.model_ini_applied\)/,
  );
});
