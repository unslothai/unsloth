// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { adoptResidentModelStatus } from "../src/features/hub/lib/adopt-inference-status.ts";
import {
  ggufVariantsMatch,
  residentModelIdMatches,
} from "../src/features/hub/lib/model-identity.ts";
import { subscribeResidentStatusRefresh } from "../src/features/hub/lib/resident-status-refresh.ts";
import { emptyStore, fakeTargets, spies } from "./helpers/kit.ts";

const RESIDENT = {
  checkpointId: "unsloth/Qwen3-8B-GGUF",
  ggufVariant: "Q4_K_M",
};

/**
 * Store actions that refuse to be called. Each message names what must not happen, so a
 * test states its rule by the action it declines to forbid.
 */
function refusing(messages: {
  setCheckpoint?: string;
  clearCheckpoint?: string;
  applyStatus?: string;
}) {
  const refuse = (message = "unreachable") => {
    return () => {
      throw new Error(message);
    };
  };
  return {
    setCheckpoint: refuse(messages.setCheckpoint),
    clearCheckpoint: refuse(messages.clearCheckpoint),
    applyStatus: refuse(messages.applyStatus),
  };
}

test("landing on the Hub synchronizes the full resident status", () => {
  // Cache mutation guards read the same resident store as Chat.
  const { calls, actions } = spies();
  const adopted = adoptResidentModelStatus(RESIDENT, emptyStore(), actions);
  assert.equal(adopted, true);
  assert.deepEqual(calls, [
    "setCheckpoint:unsloth/Qwen3-8B-GGUF:Q4_K_M",
    "applyStatus",
  ]);
});

test("a checkpoint that already matches is still hydrated", () => {
  // A reload rehydrates the checkpoint without the fields saying how it launched.
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
  // applyActiveModelStatusToStore tells a hydration from steady state by the previous
  // checkpoint/quant, so reading it after setCheckpoint keeps the old quant's baselines.
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
  // An external selection has no local runtime mirror.
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
  // The load applies its own status when it settles, and owns the params meanwhile.
  const { calls, actions } = spies();
  const adopted = adoptResidentModelStatus(
    RESIDENT,
    emptyStore({ modelLoading: true }),
    actions,
  );
  assert.equal(adopted, false);
  assert.deepEqual(calls, []);
});

test("an empty status drops the checkpoint when idle unload is disarmed", () => {
  // Nothing will bring the model back, so it really is gone.
  const cleared: string[] = [];
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    emptyStore({ checkpoint: "/models/llama.gguf", activeGgufVariant: "Q4_K_M" }),
    {
      ...refusing({
        setCheckpoint: "nothing is resident, so nothing may be pinned",
        applyStatus: "there is no status to apply",
      }),
      clearCheckpoint: () => {
        cleared.push("cleared");
      },
    },
  );
  assert.equal(adopted, true);
  assert.deepEqual(cleared, ["cleared"]);
});

test("an empty status leaves the checkpoint pinned while idle unload is armed", () => {
  // An empty status is not the model going away: an idle unload frees it but keeps a stash the
  // next request reloads, and /status cannot tell the two apart, so this must not clear it.
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    emptyStore({
      checkpoint: "/models/llama.gguf",
      activeGgufVariant: "Q4_K_M",
      idleUnloadArmed: true,
    }),
    refusing({
      setCheckpoint: "nothing is resident, so nothing may be pinned",
      clearCheckpoint: "the stash will reload exactly this model",
      applyStatus: "there is no status to apply",
    }),
  );
  assert.equal(adopted, false);
});

test("a speech model in the slot clears the pick even while idle unload is armed", () => {
  // Not the idle eviction the rule above is about: an Audio load took the single slot and
  // no stash reloads the chat model, so holding the pick left the Hub calling it Loaded.
  const cleared: string[] = [];
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null, speechOnly: true },
    emptyStore({
      checkpoint: "/models/llama.gguf",
      activeGgufVariant: "Q4_K_M",
      idleUnloadArmed: true,
    }),
    {
      ...refusing({
        setCheckpoint: "a speech model is not something chat may be pinned to",
        applyStatus: "its status describes a model chat is not talking to",
      }),
      clearCheckpoint: () => {
        cleared.push("cleared");
      },
    },
  );
  assert.equal(adopted, true);
  assert.deepEqual(cleared, ["cleared"]);
});

test("a speech model does not fight a load this tab started", () => {
  // The load owns the store until it settles, speech or not.
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null, speechOnly: true },
    emptyStore({ checkpoint: "/models/llama.gguf", modelLoading: true }),
    refusing({ clearCheckpoint: "the load owns the store until it settles" }),
  );
  assert.equal(adopted, false);
});

test("an empty status leaves an external pick alone", () => {
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    emptyStore({
      checkpoint: "gemini/gemini-2.5-pro",
      checkpointIsExternal: true,
    }),
    refusing({
      setCheckpoint: "an external pick is not a local model",
      applyStatus: "the resident GGUF belongs to another runtime",
    }),
  );
  assert.equal(adopted, false);
});

test("an empty status does not fight a load this tab started", () => {
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    emptyStore({
      checkpoint: "/models/llama.gguf",
      modelLoading: true,
    }),
    refusing({ setCheckpoint: "the load owns the store until it settles" }),
  );
  assert.equal(adopted, false);
});

test("coming back to the window re-reads inference status", () => {
  // An API request auto-switches at any time, and the Hub's only other read is its mount
  // effect, so without this the catalog keeps describing the previous model.
  const targets = fakeTargets();
  let reads = 0;
  subscribeResidentStatusRefresh(() => {
    reads += 1;
  }, targets);

  assert.equal(reads, 0, "subscribing must not read on its own");
  targets.fire("window", "focus");
  assert.equal(reads, 1);
  targets.fire("document", "visibilitychange");
  assert.equal(reads, 2);
});

test("a tab going hidden does not read", () => {
  // visibilitychange fires on the way out too, and a hidden tab has nothing to correct.
  const targets = fakeTargets();
  let reads = 0;
  subscribeResidentStatusRefresh(() => {
    reads += 1;
  }, targets);

  targets.hidden = true;
  targets.fire("document", "visibilitychange");
  assert.equal(reads, 0);

  targets.hidden = false;
  targets.fire("document", "visibilitychange");
  assert.equal(reads, 1);
});

test("an auto-switch under a mounted Hub updates the mutation guard", () => {
  const store = emptyStore({
    checkpoint: "unsloth/Qwen3-8B-GGUF",
    activeGgufVariant: "Q4_K_M",
  });
  // What the server reports once the API request has switched it.
  let serverStatus = {
    checkpointId: "unsloth/Llama-3.1-8B-Instruct-GGUF",
    ggufVariant: "Q8_0",
  };
  const readStatusAndAdopt = () => {
    adoptResidentModelStatus(
      serverStatus,
      { ...store },
      {
        setCheckpoint: (checkpointId, ggufVariant) => {
          store.checkpoint = checkpointId;
          store.activeGgufVariant = ggufVariant;
        },
        applyStatus: () => undefined,
      },
    );
  };

  const selectedModelIsResident = () =>
    residentModelIdMatches(store.checkpoint, serverStatus.checkpointId) &&
    ggufVariantsMatch(store.activeGgufVariant, serverStatus.ggufVariant);

  const targets = fakeTargets();
  subscribeResidentStatusRefresh(readStatusAndAdopt, targets);

  assert.equal(
    selectedModelIsResident(),
    false,
    "precondition: the mount-time read predates the switch",
  );
  targets.fire("window", "focus");
  assert.equal(selectedModelIsResident(), true);

  // A load this tab started owns the store until it settles, so a mid-switch refresh
  // must not re-pin the model being moved away from.
  store.modelLoading = true;
  serverStatus = {
    checkpointId: "unsloth/Qwen3-8B-GGUF",
    ggufVariant: "Q4_K_M",
  };
  targets.fire("window", "focus");
  assert.equal(store.checkpoint, "unsloth/Llama-3.1-8B-Instruct-GGUF");
});

test("unsubscribing stops the reads and leaves no listener behind", () => {
  const targets = fakeTargets();
  let reads = 0;
  const unsubscribe = subscribeResidentStatusRefresh(() => {
    reads += 1;
  }, targets);

  assert.equal(targets.listenerCount(), 2);
  unsubscribe();
  assert.equal(targets.listenerCount(), 0);
  targets.fire("window", "focus");
  targets.fire("document", "visibilitychange");
  assert.equal(reads, 0);
});
