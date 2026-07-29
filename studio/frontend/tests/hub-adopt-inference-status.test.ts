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

test("an empty status drops a local checkpoint the server no longer has", () => {
  // Unloading from another tab, from the API monitor or over the API leaves this
  // store pinned; the settings page then treats the row as resident and seeds the
  // editor from a launch config nothing is running.
  const cleared: string[] = [];
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    {
      checkpoint: "/models/llama.gguf",
      checkpointIsExternal: false,
      activeGgufVariant: "Q4_K_M",
      modelLoading: false,
    },
    {
      setCheckpoint: () => {
        throw new Error("nothing is resident, so nothing may be pinned");
      },
      clearCheckpoint: () => {
        cleared.push("cleared");
      },
      applyStatus: () => {
        throw new Error("there is no status to apply");
      },
    },
  );
  assert.equal(adopted, true);
  assert.deepEqual(cleared, ["cleared"]);
});

test("an empty status leaves an external pick alone", () => {
  // clearCheckpoint also drops the persisted external selection, so an empty
  // status must not reach it: the local model is not what the user is talking to.
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    {
      checkpoint: "gemini/gemini-2.5-pro",
      checkpointIsExternal: true,
      activeGgufVariant: null,
      modelLoading: false,
    },
    {
      setCheckpoint: () => {
        throw new Error("unreachable");
      },
      clearCheckpoint: () => {
        throw new Error("an external pick must survive an empty status");
      },
      applyStatus: () => {
        throw new Error("unreachable");
      },
    },
  );
  assert.equal(adopted, false);
});

test("an empty status does not fight a load this tab started", () => {
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    {
      checkpoint: "/models/llama.gguf",
      checkpointIsExternal: false,
      activeGgufVariant: null,
      modelLoading: true,
    },
    {
      setCheckpoint: () => {
        throw new Error("unreachable");
      },
      clearCheckpoint: () => {
        throw new Error("the load owns the store until it settles");
      },
      applyStatus: () => {
        throw new Error("unreachable");
      },
    },
  );
  assert.equal(adopted, false);
});

test("an empty status on an already empty store changes nothing", () => {
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    {
      checkpoint: null,
      checkpointIsExternal: false,
      activeGgufVariant: null,
      modelLoading: false,
    },
    {
      setCheckpoint: () => {
        throw new Error("unreachable");
      },
      clearCheckpoint: () => {
        throw new Error("there is nothing to clear");
      },
      applyStatus: () => {
        throw new Error("unreachable");
      },
    },
  );
  assert.equal(adopted, false);
});
