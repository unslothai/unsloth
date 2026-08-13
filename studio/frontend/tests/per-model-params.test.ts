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

test("an edit is filed against the model it was made for", () => {
  const next = getRememberedParamsPatch(true, {}, QWEN, { temperature: 0.2 });
  assert.deepEqual(next, { [QWEN]: { temperature: 0.2 } });
});

test("a later edit merges into the same model rather than replacing it", () => {
  const first = getRememberedParamsPatch(true, {}, QWEN, { temperature: 0.2 });
  assert.ok(first);
  const second = getRememberedParamsPatch(true, first, QWEN, {
    systemPrompt: "Be terse.",
  });
  assert.deepEqual(second, {
    [QWEN]: { temperature: 0.2, systemPrompt: "Be terse." },
  });
});

test("editing one model leaves every other model's memory alone", () => {
  const before = { [LLAMA]: { temperature: 0.9 } };
  const after = getRememberedParamsPatch(true, before, QWEN, {
    temperature: 0.2,
  });
  assert.deepEqual(after, {
    [LLAMA]: { temperature: 0.9 },
    [QWEN]: { temperature: 0.2 },
  });
  // The input map is not mutated: the store compares by identity to decide
  // whether anything has to be persisted.
  assert.deepEqual(before, { [LLAMA]: { temperature: 0.9 } });
});

// Null is how the store leaves the map and its hydration version untouched.
test("nothing is recorded when there is nothing to record", () => {
  assert.equal(
    getRememberedParamsPatch(false, {}, QWEN, { temperature: 0.2 }),
    null,
    "the feature being off records nothing",
  );
  assert.equal(
    getRememberedParamsPatch(true, {}, undefined, { temperature: 0.2 }),
    null,
    "no model selected records nothing",
  );
  assert.equal(
    getRememberedParamsPatch(true, {}, "", { temperature: 0.2 }),
    null,
    "an empty checkpoint is not a model id",
  );
  assert.equal(
    getRememberedParamsPatch(true, {}, QWEN, {}),
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
