// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// The real comparator, not a stand-in: scoping the eject to one row is the whole
// behaviour, and a permissive fake would prove nothing about it.
import { modelIdsMatch } from "../src/features/hub/lib/model-identity.ts";
import {
  type ResidentChatModel,
  ejectChatModel,
} from "../src/features/loaded-models/eject-chat-model.ts";

function resident(checkpoint: string): ResidentChatModel {
  return { checkpoint, aliases: [checkpoint] };
}

/** A backend whose resident model follows `timeline`, one entry per status read.
 *  An entry after the first is what an API auto-switch left there. */
function backend(timeline: (ResidentChatModel | null)[]) {
  const unloaded: string[] = [];
  let read = 0;
  return {
    unloaded,
    deps: {
      readResident: async () => timeline[Math.min(read++, timeline.length - 1)],
      unload: async (modelPath: string) => {
        unloaded.push(modelPath);
      },
      matches: modelIdsMatch,
    },
  };
}

test("the row's model is unloaded and reported free", async () => {
  const { unloaded, deps } = backend([resident("unsloth/Qwen3-4B"), null]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(unloaded, ["unsloth/Qwen3-4B"]);
  assert.deepEqual(result.unloadedAliases, ["unsloth/Qwen3-4B"]);
  assert.equal(result.stillResident, null);
});

// The reason this is scoped: the row is up to one poll old, so an auto-switch
// can land between the poll and the click.
test("a model that replaced the row's before the click is left alone", async () => {
  const { unloaded, deps } = backend([resident("unsloth/Llama-3.2-3B")]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.ok(
    !unloaded.includes("unsloth/Llama-3.2-3B"),
    "the model nobody clicked must survive",
  );
  // Still named directly, since the Transformers backend can cache it.
  assert.deepEqual(unloaded, ["unsloth/Qwen3-4B"]);
  assert.equal(result.stillResident, null);
});

test("a switch landing mid-eject is not chased", async () => {
  const { unloaded, deps } = backend([
    resident("unsloth/Qwen3-4B"),
    resident("unsloth/Llama-3.2-3B"),
  ]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(unloaded, ["unsloth/Qwen3-4B"]);
  assert.equal(result.stillResident, null);
});

test("a target that survives its own unload is reported still resident", async () => {
  const { unloaded, deps } = backend([resident("unsloth/Qwen3-4B")]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  // Two passes, then it gives up and names it rather than looping.
  assert.deepEqual(unloaded, ["unsloth/Qwen3-4B", "unsloth/Qwen3-4B"]);
  assert.equal(result.stillResident, "unsloth/Qwen3-4B");
});

test("a cached row with nothing resident is still unloaded by name", async () => {
  const { unloaded, deps } = backend([null]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(unloaded, ["unsloth/Qwen3-4B"]);
  assert.deepEqual(result.unloadedAliases, ["unsloth/Qwen3-4B"]);
});

test("the load path and the advertised repo id are the same row", async () => {
  const loadPath = "/models/hub/models--unsloth--Qwen3-4B/snapshots/abc";
  const { unloaded, deps } = backend([
    { checkpoint: loadPath, aliases: [loadPath, "unsloth/Qwen3-4B"] },
    null,
  ]);
  await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(unloaded, [loadPath], "matched by identity, not by string");
});
