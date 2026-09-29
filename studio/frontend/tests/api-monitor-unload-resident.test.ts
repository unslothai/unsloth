// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// The page .tsx pulls in the router, motion and hugeicons, so it cannot be imported here.
// Its Unload sequence lives in a plain module, which this drives directly.
import {
  type ResidentModel,
  type UnloadResidentDeps,
  unloadResident,
} from "../src/features/api-monitor/unload-resident.ts";

function resident(checkpoint: string, advertised?: string): ResidentModel {
  return {
    checkpoint,
    aliases: advertised ? [checkpoint, advertised] : [checkpoint],
  };
}

/** A backend whose resident model follows `timeline`, one entry consumed per status read,
 * so an entry after the first is a model an API auto-switch put there under the click.
 *
 * `unload` lands after the status read that preceded it, on whatever the switch left
 * behind, and frees VRAM only when the id it names is that model: the route treats an id
 * a concurrent load already replaced as a successful no-op (routes/inference.py:7153 ->
 * orchestrator.py:1386-1391), so a stale name returns 200 and evicts nothing. */
function backend(timeline: (ResidentModel | null)[]): UnloadResidentDeps & {
  readonly sent: string[];
  readonly reads: number;
  peek: () => string | null;
} {
  let index = 0;
  let reads = 0;
  const sent: string[] = [];
  const at = (i: number) => timeline[Math.min(i, timeline.length - 1)] ?? null;
  return {
    readResident: async () => {
      reads += 1;
      return at(index++);
    },
    unload: async (checkpoint: string) => {
      sent.push(checkpoint);
      const landsOn = at(index);
      if (landsOn && landsOn.checkpoint === checkpoint) {
        // A real eviction: nothing is resident from here on.
        timeline.splice(index, timeline.length - index, null);
      }
    },
    get sent() {
      return sent;
    },
    get reads() {
      return reads;
    },
    // What is still occupying VRAM once the run ends.
    peek: () => at(index)?.checkpoint ?? null,
  };
}

test("unloads the resident model and reports nothing left", async () => {
  const b = backend([resident("/models/a.gguf", "org/a"), null]);
  const result = await unloadResident(b);
  assert.deepEqual(b.sent, ["/models/a.gguf"]);
  assert.deepEqual(result.unloadedAliases, ["/models/a.gguf", "org/a"]);
  assert.equal(result.stillResident, null);
  assert.equal(b.peek(), null);
});

test("nothing loaded: no unload is sent", async () => {
  const b = backend([null]);
  const result = await unloadResident(b);
  assert.deepEqual(b.sent, []);
  assert.deepEqual(result.unloadedAliases, []);
  assert.equal(result.stillResident, null);
});

test("an API auto-switch under the click does not leave the new model resident", async () => {
  // The switch replaces A with B between the status read and the unload reaching the
  // lifecycle gate, so the /unload naming A is a 200 no-op. Without a recheck the button
  // reports success while B keeps the VRAM.
  const b = backend([resident("/models/a.gguf"), resident("/models/b.gguf")]);
  const result = await unloadResident(b);
  assert.equal(b.peek(), null, "B must not stay resident");
  assert.deepEqual(b.sent, ["/models/a.gguf", "/models/b.gguf"]);
  assert.deepEqual(result.unloadedAliases, [
    "/models/a.gguf",
    "/models/b.gguf",
  ]);
  assert.equal(result.stillResident, null);
});

test("a switch the recheck cannot catch is reported, not swallowed", async () => {
  // Bounded: a model that arrives after the recheck is a fresh load, not this click's
  // target, so the run names it instead of claiming a free backend.
  const b = backend([
    resident("/models/a.gguf"),
    resident("/models/b.gguf"),
    resident("/models/c.gguf"),
  ]);
  const result = await unloadResident(b);
  assert.deepEqual(b.sent, ["/models/a.gguf", "/models/b.gguf"]);
  assert.equal(result.stillResident, "/models/c.gguf");
});

test("the steady-state click costs one extra status read, never an extra unload", async () => {
  const b = backend([resident("/models/a.gguf"), null]);
  await unloadResident(b);
  assert.equal(b.sent.length, 1);
  assert.equal(b.reads, 2);
});
