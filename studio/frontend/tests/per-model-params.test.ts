// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat settings hold one set of sampling params, so switching models used to hand
// the next model the previous one's temperature and system prompt. These pin the
// rules the per-model memory follows, including the cases where it must NOT act:
// a model with nothing remembered keeps what is on screen, and an edit that moved
// nothing must not mark the map dirty.

import assert from "node:assert/strict";
import test from "node:test";

import {
  PERSISTED_INFERENCE_PARAM_KEYS,
  getRememberedParamsPatch,
  getReplayedParams,
  pickPersistedInferenceParams,
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

/** What the store passes: the edit that moved, plus the full resulting params. */
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
    pickPersistedInferenceParams(full),
  );
}

test("an edit is filed against the model it was made for", () => {
  const next = record({}, QWEN, { temperature: 0.2 }, params({ temperature: 0.2 }));
  assert.equal(next?.[QWEN].temperature, 0.2);
});

// The entry has to be the whole snapshot. Replay overlays it onto the outgoing
// model's params, so anything missing would silently keep the other model's value.
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
  const after = record(before, QWEN, { temperature: 0.2 }, params({ temperature: 0.2 }));
  assert.deepEqual(after?.[LLAMA], { temperature: 0.9 });
  assert.equal(after?.[QWEN].temperature, 0.2);
  // The input map is not mutated: the store compares by identity to decide
  // whether anything has to be persisted.
  assert.deepEqual(before, { [LLAMA]: { temperature: 0.9 } });
});

// The reported failure: A remembered only its temperature, so switching back
// overlaid it onto B's prompt and A ran under settings it was never given.
test("a model returns to the prompt it was last used with", () => {
  const aParams = params({ checkpoint: QWEN, temperature: 0.2, systemPrompt: "A" });
  const memory = record({}, QWEN, { temperature: 0.2 }, aParams);
  assert.ok(memory);

  // Switch to B, which has no memory, then change only B's prompt.
  const onB = getReplayedParams(true, memory, aParams, LLAMA, true);
  const bParams = { ...onB, checkpoint: LLAMA, systemPrompt: "B" };
  const afterB = record(memory, LLAMA, { systemPrompt: "B" }, bParams);
  assert.ok(afterB);

  const backOnA = getReplayedParams(true, afterB, bParams, QWEN, true);
  assert.equal(backOnA.systemPrompt, "A");
  assert.equal(backOnA.temperature, 0.2);
});

// The interactive local load never calls setCheckpoint first: it calls
// setParams with the destination checkpoint and the backend's recommended
// params. Replay has to happen on that call or the common switch restores
// nothing. This walks that exact sequence through the two helpers setParams uses.
test("a local model load replays memory over the backend's recommendation", () => {
  let memory: Record<string, ReturnType<typeof pickPersistedInferenceParams>> = {};
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
  assert.equal(live.temperature, 0.6, "a model with no memory takes its recommendation");

  loadModel(QWEN, { temperature: 0.7 });
  assert.equal(live.temperature, 0.2, "the tuned value beats the recommendation");
});

// Null is how the store leaves the map and its hydration version untouched.
test("nothing is recorded when there is nothing to record", () => {
  const snapshot = pickPersistedInferenceParams(params());
  assert.equal(
    getRememberedParamsPatch(false, {}, QWEN, { temperature: 0.2 }, snapshot),
    null,
    "the feature being off records nothing",
  );
  assert.equal(
    getRememberedParamsPatch(true, {}, undefined, { temperature: 0.2 }, snapshot),
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
  // Params the model never pinned carry over rather than snapping to defaults.
  assert.equal(replayed.topP, current.topP);
});

test("a model with nothing remembered keeps the settings on screen", () => {
  const current = params({ temperature: 0.33 });
  const replayed = getReplayedParams(true, {}, current, LLAMA, true);
  assert.equal(replayed, current, "returned by identity, so nothing persists");
});

test("re-selecting the same model does not replay over a live edit", () => {
  const current = params({ temperature: 0.33 });
  // checkpointChanged=false: the user just nudged a slider, and replaying the
  // stored value here would undo the edit they are making.
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

// Turning the setting on adopts what is on screen for the active model, so the
// first switch away and back returns to it rather than to nothing.
test("the persisted snapshot covers every persisted key and excludes the checkpoint", () => {
  const picked = pickPersistedInferenceParams(params());
  assert.equal(
    "checkpoint" in picked,
    false,
    "the checkpoint names the model, it is not one of the values",
  );
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    if (params()[key] !== undefined) {
      assert.ok(key in picked, `${key} should be captured`);
    }
  }
});

test("the snapshot drops params the current model never set", () => {
  const picked = pickPersistedInferenceParams(
    params({ topK: undefined as unknown as number }),
  );
  assert.equal("topK" in picked, false);
});
