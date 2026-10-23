// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { InvokeStreamEvent as Event } from "../src/features/flows/player/player-api.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const L = await import("../src/features/flows/player/world/live-reply.ts");
const A = await import("../src/features/flows/player/player-api.ts");

async function* events(list: Event[]): AsyncGenerator<Event> {
  for (const e of list) yield e;
}

test("an NPC line grows as its pieces arrive", () => {
  let live = L.appendReplyDelta(null, "tom", 0, 0, "Hel");
  live = L.appendReplyDelta(live, "tom", 0, 0, "lo");
  assert.deepEqual(live, { npcId: "tom", generation: 0, text: "Hello" });
  // Another villager's reply starts a fresh line.
  assert.equal(L.appendReplyDelta(live, "fern", 0, 0, "Oh")?.text, "Oh");
});

test("pieces of a reply from before a save load are dropped", () => {
  const live = L.appendReplyDelta(null, "tom", 1, 1, "New game");
  assert.equal(L.appendReplyDelta(live, "tom", 0, 1, " old piece"), live);
  assert.equal(L.appendReplyDelta(null, "tom", 0, 1, "stale"), null);
});

test("a streamed answer hands over its text, then returns the outputs", async () => {
  const seen: string[] = [];
  const result = await A.consumeInvokeStream(
    events([
      { type: "delta", port: "reply", text: "Hi " },
      { type: "delta", port: "reply", text: "there" },
      { type: "result", outputs: { reply: { text: "Hi there" } }, rev: 2 },
    ]),
    (text) => seen.push(text),
  );
  assert.deepEqual(seen, ["Hi ", "there"]);
  assert.deepEqual(result.outputs, { reply: { text: "Hi there" } });
  assert.equal(result.rev, 2);
});

test("a streamed answer that fails or stops early is an error", async () => {
  await assert.rejects(
    A.consumeInvokeStream(
      events([{ type: "error", status: 409, detail: "The model is busy" }]),
      () => undefined,
    ),
    /busy/,
  );
  await assert.rejects(
    A.consumeInvokeStream(
      events([{ type: "delta", text: "Hm" }]),
      () => undefined,
    ),
    /stopped before it finished/,
  );
});
