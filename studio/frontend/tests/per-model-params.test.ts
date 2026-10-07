// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// preset-policy imports extensionless, the way vite resolves.
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

import {
  REMEMBERED_INFERENCE_PARAM_KEYS,
  getRememberedParamsPatch,
  getReplayedParams,
  pickRememberedParams,
} from "../src/features/chat/lib/per-model-params.ts";
import type { InferenceParams } from "../src/features/chat/types/runtime.ts";

const QWEN = "unsloth/Qwen3.5-9B-GGUF";
const LLAMA = "unsloth/Llama-4-8B";

function params(overrides: Partial<InferenceParams> = {}): InferenceParams {
  return {
    checkpoint: QWEN,
    temperature: 0.7,
    topP: 0.95,
    topK: 40,
    minP: 0.05,
    repetitionPenalty: 1,
    presencePenalty: 0,
    maxSeqLength: 4096,
    maxTokens: 2048,
    systemPrompt: "",
    systemVariables: "",
    ...overrides,
  } as InferenceParams;
}

function record(
  paramsByModel: Record<string, Record<string, unknown>>,
  modelId: string | undefined,
  changed: Record<string, unknown>,
  full: InferenceParams,
) {
  return getRememberedParamsPatch(
    true,
    paramsByModel as never,
    modelId,
    changed as never,
    pickRememberedParams(full),
  );
}

test("an edit is filed against the model it was made for", () => {
  const next = record(
    {},
    QWEN,
    { temperature: 0.2 },
    params({ temperature: 0.2 }),
  );
  assert.equal(next?.[QWEN].temperature, 0.2);
});

// Replay overlays onto the outgoing model's params, so missing keys leak across models.
test("an entry records every param, not only the one that moved", () => {
  const next = record(
    {},
    QWEN,
    { temperature: 0.2 },
    params({ temperature: 0.2, systemPrompt: "Be terse." }),
  );
  assert.equal(next?.[QWEN].systemPrompt, "Be terse.");
  assert.equal(next?.[QWEN].topP, 0.95);
});

test("editing one model leaves every other model's memory alone", () => {
  const before = { [LLAMA]: { temperature: 0.9 } };
  const after = record(
    before,
    QWEN,
    { temperature: 0.2 },
    params({ temperature: 0.2 }),
  );
  assert.deepEqual(after?.[LLAMA], { temperature: 0.9 });
  assert.equal(after?.[QWEN].temperature, 0.2);
  // The input map is not mutated: the store compares by identity to decide
  // whether anything has to be persisted.
  assert.deepEqual(before, { [LLAMA]: { temperature: 0.9 } });
});

test("a model returns to the prompt it was last used with", () => {
  const aParams = params({
    checkpoint: QWEN,
    temperature: 0.2,
    systemPrompt: "A",
  });
  const memory = record({}, QWEN, { temperature: 0.2 }, aParams);
  assert.ok(memory);

  const onB = getReplayedParams(true, memory, aParams, LLAMA, true);
  const bParams = { ...onB, checkpoint: LLAMA, systemPrompt: "B" };
  const afterB = record(memory, LLAMA, { systemPrompt: "B" }, bParams);
  assert.ok(afterB);

  const backOnA = getReplayedParams(true, afterB, bParams, QWEN, true);
  assert.equal(backOnA.systemPrompt, "A");
  assert.equal(backOnA.temperature, 0.2);
});

// Local load calls setParams with the destination checkpoint, not setCheckpoint.
test("a local model load replays memory over the backend's recommendation", () => {
  let memory: Record<string, ReturnType<typeof pickRememberedParams>> = {};
  let live = params({ checkpoint: QWEN });

  function loadModel(modelId: string, recommended: Partial<InferenceParams>) {
    const merged = { ...live, ...recommended, checkpoint: modelId };
    const replayed = getReplayedParams(true, memory, merged, modelId, true);
    memory = record(memory, modelId, { temperature: 1 }, replayed) ?? memory;
    live = replayed;
  }

  function edit(overrides: Partial<InferenceParams>) {
    live = { ...live, ...overrides };
    memory = record(memory, live.checkpoint, overrides, live) ?? memory;
  }

  loadModel(QWEN, { temperature: 0.7 });
  edit({ temperature: 0.2 });
  loadModel(LLAMA, { temperature: 0.6 });
  assert.equal(
    live.temperature,
    0.6,
    "a model with no memory takes its recommendation",
  );

  loadModel(QWEN, { temperature: 0.7 });
  assert.equal(
    live.temperature,
    0.2,
    "the tuned value beats the recommendation",
  );
});

test("the model being switched away from is remembered", () => {
  const onA = params({ checkpoint: QWEN, temperature: 0.2, systemPrompt: "A" });
  const memory = record({}, QWEN, pickRememberedParams(onA), onA);
  assert.ok(memory, "leaving A records what A was running with");

  const onB = getReplayedParams(true, memory, onA, LLAMA, true);
  const editedB = { ...onB, checkpoint: LLAMA, systemPrompt: "B" };
  const afterB =
    record(memory, LLAMA, { systemPrompt: "B" }, editedB) ?? memory;

  const backOnA = getReplayedParams(true, afterB, editedB, QWEN, true);
  assert.equal(backOnA.systemPrompt, "A");
  assert.equal(backOnA.temperature, 0.2);
});

test("a load response does not overwrite a remembered token budget", () => {
  const memory = {
    [QWEN]: { ...pickRememberedParams(params()), maxTokens: 4096 },
  };
  const afterCheckpoint = getReplayedParams(true, memory, params(), QWEN, true);
  assert.equal(afterCheckpoint.maxTokens, 4096);

  const loadResponse = { ...afterCheckpoint, maxTokens: 131072 };
  const withoutForcedReplay = getReplayedParams(
    true,
    memory,
    loadResponse,
    QWEN,
    false,
  );
  assert.equal(
    withoutForcedReplay.maxTokens,
    131072,
    "this is the reported bug",
  );

  const withForcedReplay = getReplayedParams(
    true,
    memory,
    loadResponse,
    QWEN,
    true,
  );
  assert.equal(withForcedReplay.maxTokens, 4096);
});

test("a model's defaults do not outrank what it is remembered with", async () => {
  const { mergeBackendRecommendedInference } = await import(
    "../src/features/chat/presets/preset-policy.ts"
  );
  const response = {
    inference: { temperature: 0.9, top_p: 0.5 },
    is_gguf: true,
    context_length: 131072,
  };
  const applyDefaults = (current: InferenceParams, modelId: string) =>
    mergeBackendRecommendedInference({
      current,
      response: response as never,
      modelId,
      presetSource: "builtin-default",
      loadedContextLength: response.context_length,
    });
  const memory = { [QWEN]: { temperature: 0.2, maxTokens: 4096 } };

  const tuned = applyDefaults(
    params({ temperature: 0.2, maxTokens: 4096 }),
    QWEN,
  );
  assert.equal(tuned.temperature, 0.9, "this is the reported clobber");
  const replayed = getReplayedParams(true, memory, tuned, QWEN, true);
  assert.equal(replayed.temperature, 0.2);
  assert.equal(replayed.maxTokens, 4096);

  const fresh = getReplayedParams(
    true,
    memory,
    applyDefaults(params(), LLAMA),
    LLAMA,
    true,
  );
  assert.equal(
    fresh.temperature,
    0.9,
    "a model with no memory keeps its own defaults",
  );
});

test("nothing is recorded when there is nothing to record", () => {
  const snapshot = pickRememberedParams(params());
  assert.equal(
    getRememberedParamsPatch(false, {}, QWEN, { temperature: 0.2 }, snapshot),
    null,
    "the feature being off records nothing",
  );
  assert.equal(
    getRememberedParamsPatch(
      true,
      {},
      undefined,
      { temperature: 0.2 },
      snapshot,
    ),
    null,
    "no model selected records nothing",
  );
  assert.equal(
    getRememberedParamsPatch(true, {}, "", { temperature: 0.2 }, snapshot),
    null,
    "an empty checkpoint is not a model id",
  );
  assert.equal(
    getRememberedParamsPatch(true, {}, QWEN, {}, snapshot),
    null,
    "an edit that moved no persisted param records nothing",
  );
});

test("switching models replays that model's own settings", () => {
  const current = params({ temperature: 0.7, systemPrompt: "" });
  const replayed = getReplayedParams(
    true,
    { [LLAMA]: { temperature: 0.1, systemPrompt: "Be terse." } },
    current,
    LLAMA,
    true,
  );
  assert.equal(replayed.temperature, 0.1);
  assert.equal(replayed.systemPrompt, "Be terse.");
  assert.equal(replayed.topP, current.topP);
});

test("a model with nothing remembered keeps the settings on screen", () => {
  const current = params({ temperature: 0.33 });
  const replayed = getReplayedParams(true, {}, current, LLAMA, true);
  assert.equal(replayed, current, "returned by identity, so nothing persists");
});

test("re-selecting the same model does not replay over a live edit", () => {
  const current = params({ temperature: 0.33 });
  const replayed = getReplayedParams(
    true,
    { [QWEN]: { temperature: 0.9 } },
    current,
    QWEN,
    false,
  );
  assert.equal(replayed, current);
});

test("the feature being off leaves a model switch alone", () => {
  const current = params({ temperature: 0.33 });
  const replayed = getReplayedParams(
    false,
    { [LLAMA]: { temperature: 0.9 } },
    current,
    LLAMA,
    true,
  );
  assert.equal(replayed, current);
});

test("the snapshot covers every remembered key and excludes the checkpoint", () => {
  const picked = pickRememberedParams(params());
  assert.equal(
    "checkpoint" in picked,
    false,
    "the checkpoint names the model, it is not one of the values",
  );
  // The context a model loads with is already kept per model by its load
  // config, and that is the copy the load uses.
  assert.equal("maxSeqLength" in picked, false);
  for (const key of REMEMBERED_INFERENCE_PARAM_KEYS) {
    if (params()[key] !== undefined) {
      assert.ok(key in picked, `${key} should be captured`);
    }
  }
});

test("the snapshot drops params the current model never set", () => {
  const picked = pickRememberedParams(
    params({ topK: undefined as unknown as number }),
  );
  assert.equal("topK" in picked, false);
});

test("a maxSeqLength in a stored entry is not replayed over the loaded context", () => {
  const replayed = getReplayedParams(
    true,
    { [QWEN]: { temperature: 0.2, maxSeqLength: 131072 } },
    params({ checkpoint: QWEN, maxSeqLength: 4096 }),
    QWEN,
    true,
  );
  assert.equal(replayed.temperature, 0.2, "the remembered value still replays");
  assert.equal(replayed.maxSeqLength, 4096);
});

test("a key that is not an inference param cannot reach the live params", () => {
  const replayed = getReplayedParams(
    true,
    { [QWEN]: { temperature: 0.2, notAParam: 9 } as never },
    params({ checkpoint: QWEN }),
    QWEN,
    true,
  );
  assert.equal(replayed.temperature, 0.2);
  // params flows on into request bodies and into an extra="forbid" settings write.
  assert.equal("notAParam" in replayed, false);
});

for (const [label, id] of [
  ["a Windows drive path", "C:\\Users\\Daniel\\models\\Qwen3-8B"],
  ["a UNC share path", "\\\\fileserver\\models\\gemma-3-270m-it"],
  ["a WSL UNC path", "\\\\wsl$\\Ubuntu\\home\\d\\models\\llama"],
  [
    "a macOS path with spaces",
    "/Users/d/Library/Application Support/unsloth/gemma",
  ],
  [
    "a Linux cache path",
    "/home/d/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B",
  ],
  ["a provider-qualified id", "external::anthropic::claude-opus-5"],
  ["a non-ASCII repo id", "unsloth/通義千問-7B"],
] as const) {
  test(`${label} is remembered and replayed unchanged`, () => {
    const recorded = record(
      {},
      id,
      { temperature: 0.2 },
      params({ checkpoint: id, temperature: 0.2 }),
    );
    assert.ok(recorded, "nothing was recorded");
    assert.equal(recorded[id].temperature, 0.2);
    const replayed = getReplayedParams(
      true,
      recorded,
      params({ temperature: 1 }),
      id,
      true,
    );
    assert.equal(replayed.temperature, 0.2);
  });
}
