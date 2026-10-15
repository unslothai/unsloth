// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// Uses the real comparator: a permissive fake would prove nothing.
import { modelIdsMatch } from "../src/features/hub/lib/model-identity.ts";
import {
  type ResidentChatModel,
  ejectChatModel,
} from "../src/features/loaded-models/eject-chat-model.ts";

function resident(checkpoint: string): ResidentChatModel {
  return { checkpoint, aliases: [checkpoint] };
}

/** Resident model follows `timeline`, one entry per status read. */
function backend(
  timeline: (ResidentChatModel | null)[],
  cachedRow = false,
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

// The row is up to one poll old, so an auto-switch can land before the click.
test("a model that replaced the row's before the click is left alone", async () => {
  const { unloaded, deps } = backend([resident("unsloth/Llama-3.2-3B")]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.ok(
    !unloaded.includes("unsloth/Llama-3.2-3B"),
    "the model nobody clicked must survive",
  );
  // /unload answers 200 even for a model it does not hold.
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
  assert.deepEqual(unloaded, ["unsloth/Qwen3-4B", "unsloth/Qwen3-4B"]);
  assert.equal(result.stillResident, "unsloth/Qwen3-4B");
  // Callers must key the picker clear on stillResident, not the aliases.
  assert.ok(result.unloadedAliases.length > 0);
});

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

// /unload answers 200 for names it no longer holds, so success needs a re-read.
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

/** `loaded` follows `timeline`, one entry per cached read; `active` stays resident. */
function replacedBackend(
  active: ResidentChatModel | null,
  timeline: string[][],
) {
  const unloaded: string[] = [];
  let read = 0;
  return {
    unloaded,
    deps: {
      readResident: async () => active,
      unload: async (modelPath: string) => {
        unloaded.push(modelPath);
      },
      matches: modelIdsMatch,
      readCached: async () => timeline[Math.min(read++, timeline.length - 1)],
    },
  };
}

// A replacement does not evict the previous model from the registry.
test("a row replaced while still cached is unloaded, not written off", async () => {
  const { unloaded, deps } = replacedBackend(resident("unsloth/Llama-3.2-3B"), [
    ["unsloth/Qwen3-4B", "unsloth/Llama-3.2-3B"],
    ["unsloth/Llama-3.2-3B"],
  ]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(
    unloaded,
    ["unsloth/Qwen3-4B"],
    "the memory the click asked for is the memory released",
  );
  assert.deepEqual(result.unloadedAliases, ["unsloth/Qwen3-4B"]);
  assert.equal(result.stillResident, null);
  assert.equal(
    result.replacedBy,
    null,
    "an eject that ran is not a row that had already gone",
  );
});

test("a row replaced and really gone is still left alone", async () => {
  const { unloaded, deps } = replacedBackend(resident("unsloth/Llama-3.2-3B"), [
    ["unsloth/Llama-3.2-3B"],
  ]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(unloaded, [], "nothing but the row's own model may go");
  assert.deepEqual(result.unloadedAliases, []);
  assert.equal(result.replacedBy, "unsloth/Llama-3.2-3B");
});

test("an idle runtime still holding the row releases it", async () => {
  const { unloaded, deps } = replacedBackend(null, [["unsloth/Qwen3-4B"], []]);
  const result = await ejectChatModel("unsloth/Qwen3-4B", deps);
  assert.deepEqual(unloaded, ["unsloth/Qwen3-4B"]);
  assert.deepEqual(result.unloadedAliases, ["unsloth/Qwen3-4B"]);
  assert.equal(result.stillResident, null);
  assert.equal(result.replacedBy, null);
});
