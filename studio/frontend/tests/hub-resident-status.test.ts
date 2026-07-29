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
 * Store actions that refuse to be called. The message on each names what must
 * not happen, so a test states its rule by the action it declines to forbid.
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
    emptyStore({
      checkpoint: "/models/llama.gguf",
      activeGgufVariant: "Q4_K_M",
    }),
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

test("an empty status leaves an external pick alone", () => {
  // clearCheckpoint also drops the persisted external selection, so an empty
  // status must not reach it: the local model is not what the user is talking to.
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    emptyStore({
      checkpoint: "gemini/gemini-2.5-pro",
      checkpointIsExternal: true,
    }),
    refusing({
      clearCheckpoint: "an external pick must survive an empty status",
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
    refusing({ clearCheckpoint: "the load owns the store until it settles" }),
  );
  assert.equal(adopted, false);
});

test("an empty status on an already empty store changes nothing", () => {
  const adopted = adoptResidentModelStatus(
    { checkpointId: null, ggufVariant: null },
    emptyStore(),
    refusing({ clearCheckpoint: "there is nothing to clear" }),
  );
  assert.equal(adopted, false);
});

test("coming back to the window re-reads inference status", () => {
  // An OpenAI-compatible request auto-switches the resident model whenever it
  // likes. The Hub's only other status read is its mount effect, so without this
  // the catalog and the settings page keep describing the previous model for as
  // long as the Hub stays mounted.
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
  // visibilitychange fires on the way out too, and a hidden tab has no settings
  // page to correct.
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

test("an auto-switch under a mounted Hub stops hiding the live config", () => {
  // The whole point, end to end: while the Hub is mounted an OpenAI-compatible
  // request swaps the resident model. Without a second read the store still names
  // the old one, so hub-page's settingsTargetIsResident says the newly loaded
  // model is not resident, its settings page is handed loadedConfig=null, and
  // ModelConfigPage seeds the editor from saved/default values -- which Apply then
  // reloads the model with, over what the API actually selected.
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

  // hub-page.tsx's settingsTargetIsResident, for the model the API just loaded.
  const settingsTargetIsResident = () =>
    residentModelIdMatches(store.checkpoint, serverStatus.checkpointId) &&
    ggufVariantsMatch(store.activeGgufVariant, serverStatus.ggufVariant);

  const targets = fakeTargets();
  subscribeResidentStatusRefresh(readStatusAndAdopt, targets);

  assert.equal(
    settingsTargetIsResident(),
    false,
    "precondition: the mount-time read predates the switch",
  );
  targets.fire("window", "focus");
  assert.equal(settingsTargetIsResident(), true);

  // A load this tab started owns the store until it settles, so a refresh landing
  // mid-switch must not re-pin the model the user is moving away from.
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
