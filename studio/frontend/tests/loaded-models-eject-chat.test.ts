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
function backend(
  timeline: (ResidentChatModel | null)[],
  cachedRow = false,
  /** What the runtime still holds after a cached row's unload. */
  cachedAfter: string[] | null = null,
) {
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
      cachedRow,
      ...(cachedAfter === null ? {} : { readCached: async () => cachedAfter }),
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
  // Nothing at all is unloaded. /unload naming a model the backend does not
  // hold answers 200 "unloaded", so firing it would report an eject that never
  // happened and clear the picker off the back of it.
  assert.deepEqual(unloaded, []);
  assert.deepEqual(result.unloadedAliases, []);
  assert.equal(result.stillResident, null);
  assert.equal(
    result.replacedBy,
    "unsloth/Llama-3.2-3B",
    "the caller needs the replacement's name to say what took its place",
  );
});

test("an idle runtime reports the row already gone, not a fresh eject", async () => {
  const { unloaded, deps } = backend([null]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(unloaded, [], "nothing is resident, so nothing to unload");
  assert.deepEqual(result.unloadedAliases, []);
  assert.equal(result.stillResident, null);
  // Null rather than a name: the runtime holds nothing, so there is no
  // replacement to name and the row is simply stale.
  assert.equal(result.replacedBy, null);
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
  // Both are set together here, so the caller must key the picker clear on
  // stillResident: the aliases alone would empty it while the model still runs.
  assert.ok(result.unloadedAliases.length > 0);
});

// A row the backend kept past the active model is never what a status read
// reports, so it is the one case that has to be named directly.
test("a cached row with nothing resident is still unloaded by name", async () => {
  const { unloaded, deps } = backend([null], true);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(unloaded, ["unsloth/Qwen3-4B"]);
  assert.deepEqual(result.unloadedAliases, ["unsloth/Qwen3-4B"]);
  assert.equal(result.replacedBy, null);
});

test("a cached row is unloaded even while another model is active", async () => {
  const { unloaded, deps } = backend([resident("unsloth/Llama-3.2-3B")], true);
  await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(
    unloaded,
    ["unsloth/Qwen3-4B"],
    "the cached copy goes, the active model stays",
  );
});

// /unload answers 200 for a name the backend no longer holds, and the cached
// row is the one path with no scoped read to catch that, so the reported
// success was the call itself rather than any evidence of a release.
test("a cached row the backend kept is reported still resident", async () => {
  const { unloaded, deps } = backend([null], true, ["unsloth/Qwen3-4B"]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(unloaded, ["unsloth/Qwen3-4B"], "the unload was attempted");
  assert.equal(result.stillResident, "unsloth/Qwen3-4B");
  assert.deepEqual(result.unloadedAliases, [], "nothing to clear the picker on");
});

test("a cached row the backend released is reported ejected", async () => {
  const { deps } = backend([null], true, ["unsloth/Llama-3.2-3B"]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.equal(result.stillResident, null);
  assert.deepEqual(result.unloadedAliases, ["unsloth/Qwen3-4B"]);
});

// Both the row and the confirmation come from the same `loaded` list, so the
// names line up by construction; the comparator is there for the day they do not.
test("a backend that cannot be re-read leaves the old reading alone", async () => {
  const { unloaded, deps } = backend([null], true);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(unloaded, ["unsloth/Qwen3-4B"]);
  assert.deepEqual(result.unloadedAliases, ["unsloth/Qwen3-4B"]);
  assert.equal(result.stillResident, null);
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
