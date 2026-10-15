// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// The page .tsx cannot be imported here, so its Unload sequence is tested via its plain module.
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

/**
 * A backend whose resident model follows `timeline`, one entry per status read. Unloading an id
 * that a concurrent load already replaced returns 200 and evicts nothing.
 */
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
        timeline.splice(index, timeline.length - index, null);
      }
    },
    get sent() {
      return sent;
    },
    get reads() {
      return reads;
    },
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
  // A switch between the status read and /unload makes the unload a 200 no-op, so it must recheck.
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
