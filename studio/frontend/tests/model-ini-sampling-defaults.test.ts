// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// On the Default preset the Qwen3 thinking table still applies, and an unsloth.ini outranks it only for the keys it set.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { mergeBackendRecommendedInference, qwenThinkingParamsWithModelIni } =
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
    model_ini_sampling_keys?: string[];
    inference: Record<string, number>;
  },
  current = DEFAULT_INFERENCE_PARAMS,
) {
  const params = mergeBackendRecommendedInference({
    current: { ...current, checkpoint: QWEN },
    response: { is_gguf: true, ...response },
    modelId: QWEN,
    presetSource: "builtin-default",
    loadedContextLength: 4096,
  });
  const qwen = resolveQwenThinkingParams(QWEN, true);
  assert.ok(qwen);
  return { ...params, ...qwenThinkingParamsWithModelIni(qwen, response) };
}

test("every sampling key the INI set wins over the Qwen3 table", () => {
  const params = afterLoad({
    model_ini_sampling_keys: ["temperature", "top_p", "top_k", "min_p"],
    inference: { temperature: 0.42, top_p: 0.77, top_k: 17, min_p: 0.03 },
  });
  assert.equal(params.temperature, 0.42);
  assert.equal(params.topP, 0.77);
  assert.equal(params.topK, 17);
  assert.equal(params.minP, 0.03);
});

test("an INI that sets only temperature keeps the table's other values", () => {
  const params = afterLoad({
    model_ini_sampling_keys: ["temperature"],
    inference: { temperature: 0.42, top_p: 0.8, top_k: 40 },
  });
  assert.equal(params.temperature, 0.42);
  assert.equal(params.topP, 0.95);
  assert.equal(params.topK, 20);
});

test("no INI keys, or a server without the field, leaves the Qwen3 table untouched", () => {
  for (const keys of [[], undefined]) {
    const params = afterLoad({
      ...(keys ? { model_ini_sampling_keys: keys } : {}),
      inference: { temperature: 0.7, top_p: 0.8, top_k: 40 },
    });
    assert.equal(params.temperature, 0.6);
    assert.equal(params.topP, 0.95);
    assert.equal(params.topK, 20);
  }
});

test("an INI repeat-penalty reaches the slider; a load without one leaves it alone", () => {
  const withIni = afterLoad({
    model_ini_sampling_keys: ["temperature", "repetition_penalty"],
    inference: { temperature: 0.42, repetition_penalty: 1.1 },
  });
  assert.equal(withIni.repetitionPenalty, 1.1);
  const kept = afterLoad(
    { inference: { temperature: 0.7 } },
    { ...DEFAULT_INFERENCE_PARAMS, repetitionPenalty: 1.25 },
  );
  assert.equal(kept.repetitionPenalty, 1.25);
});

test("a key the file named but the response lacks, or an unknown key, changes nothing", () => {
  const qwen = { temperature: 0.6, topP: 0.95 };
  assert.deepEqual(
    qwenThinkingParamsWithModelIni(qwen, {
      model_ini_sampling_keys: ["top_p", "seed"],
      inference: { temperature: 0.42 },
    }),
    qwen,
  );
});

test("both sites that lay the thinking table over a load apply the INI's keys on top", () => {
  assert.match(
    readSrc("features/chat/hooks/use-chat-model-runtime.ts"),
    /const p =\s*qwenTable && qwenThinkingParamsWithModelIni\(qwenTable, loadResponse\);/,
  );
  assert.match(
    readSrc("features/chat/lib/apply-inference-status-to-store.ts"),
    /qwenThinkingParamsWithModelIni\(qwenParams, status\)/,
  );
});
