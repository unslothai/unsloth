// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type ResidentAdoptionState,
  adoptResidentModelStatus,
} from "../src/features/hub/lib/adopt-inference-status.ts";

const RESIDENT = {
  checkpointId: "unsloth/Qwen3-8B-GGUF",
  ggufVariant: "Q4_K_M",
};

function emptyStore(
  overrides: Partial<ResidentAdoptionState> = {},
): ResidentAdoptionState {
  return {
    checkpoint: null,
    checkpointIsExternal: false,
    activeGgufVariant: null,
    modelLoading: false,
    ...overrides,
  };
}

function spies() {
  const calls: string[] = [];
  const previouslySeen: { checkpoint: string | null; ggufVariant: string | null }[] =
    [];
  return {
    calls,
    previouslySeen,
    actions: {
      setCheckpoint(checkpointId: string, ggufVariant: string | null) {
        calls.push(`setCheckpoint:${checkpointId}:${ggufVariant ?? ""}`);
      },
      applyStatus(previous: {
        checkpoint: string | null;
        ggufVariant: string | null;
      }) {
        calls.push("applyStatus");
        previouslySeen.push(previous);
      },
    },
  };
}

test("landing on the Hub applies the whole status, not just the checkpoint", () => {
  // Nothing else on /hub hydrates the runtime store: useChatModelRuntime has no
  // mount sync and the chat page is a different route. Pinning only the
  // checkpoint leaves every field useActiveModelConfig reads at its default, so
  // the settings page offers those defaults as the resident model's live config.
  const { calls, actions } = spies();
  const adopted = adoptResidentModelStatus(RESIDENT, emptyStore(), actions);
  assert.equal(adopted, true);
  assert.deepEqual(calls, [
    "setCheckpoint:unsloth/Qwen3-8B-GGUF:Q4_K_M",
    "applyStatus",
  ]);
});

test("a checkpoint that already matches is still hydrated", () => {
  // A reload rehydrates params.checkpoint from localStorage on its own, with
  // none of the fields that say how the model was actually launched.
  const { calls, actions } = spies();
  adoptResidentModelStatus(
    RESIDENT,
    emptyStore({
      checkpoint: "unsloth/Qwen3-8B-GGUF",
      activeGgufVariant: "Q4_K_M",
    }),
    actions,
  );
  assert.deepEqual(calls, ["applyStatus"]);
});

test("an API auto-switch under the tab re-pins the model and the quant", () => {
  for (const stale of [
    { checkpoint: "unsloth/Llama-3.1-8B-GGUF", activeGgufVariant: "Q4_K_M" },
    { checkpoint: "unsloth/Qwen3-8B-GGUF", activeGgufVariant: "Q8_0" },
  ]) {
    const { calls, actions } = spies();
    adoptResidentModelStatus(RESIDENT, emptyStore(stale), actions);
    assert.deepEqual(calls, [
      "setCheckpoint:unsloth/Qwen3-8B-GGUF:Q4_K_M",
      "applyStatus",
    ]);
  }
});

test("the status applied is the one from before the checkpoint moved", () => {
  // applyActiveModelStatusToStore tells a hydration from steady state by the
  // previous checkpoint/quant, so it has to be read before setCheckpoint syncs
  // them, or a variant-only switch reads as steady state and keeps the old
  // quant's baselines.
  const { previouslySeen, actions } = spies();
  adoptResidentModelStatus(
    RESIDENT,
    emptyStore({
      checkpoint: "unsloth/Qwen3-8B-GGUF",
      activeGgufVariant: "Q8_0",
    }),
    actions,
  );
  assert.deepEqual(previouslySeen, [
    { checkpoint: "unsloth/Qwen3-8B-GGUF", ggufVariant: "Q8_0" },
  ]);
});

test("nothing is adopted when no model is loaded", () => {
  const { calls, actions } = spies();
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    emptyStore(),
    actions,
  );
  assert.equal(adopted, false);
  assert.deepEqual(calls, []);
});

test("an external-provider selection is left alone", () => {
  // It has no local mirror, so stamping the resident GGUF's launch settings onto
  // it would describe a model the user is not talking to.
  const { calls, actions } = spies();
  const adopted = adoptResidentModelStatus(
    RESIDENT,
    emptyStore({
      checkpoint: "openai/gpt-5",
      checkpointIsExternal: true,
    }),
    actions,
  );
  assert.equal(adopted, false);
  assert.deepEqual(calls, []);
});

test("a load in flight is not fought", () => {
  // The load applies its own status when it settles, and the load dialog owns
  // the params meanwhile.
  const { calls, actions } = spies();
  const adopted = adoptResidentModelStatus(
    RESIDENT,
    emptyStore({ modelLoading: true }),
    actions,
  );
  assert.equal(adopted, false);
  assert.deepEqual(calls, []);
});
