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

test("loading a save stops the reply still being written", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL(
      "../src/features/flows/player/world/world-player.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(src, /signal: run\.signal,\s*onDelta:/);
  assert.match(
    src,
    /cancelLiveWork\(loads, messages, \[replyRun, summaryRun, directorRun\]\);\s*queue\.clear\(\);/,
  );
});

test("leaving the world stops the pending reply and its follow-up model calls", async () => {
  const loads = { current: 3 };
  const messages = { current: 7 };
  const reply = { current: new AbortController() as AbortController | null };
  const summary = { current: new AbortController() as AbortController | null };
  const director = { current: null as AbortController | null };
  const signals = [reply.current!.signal, summary.current!.signal];
  const turn = messages.current;
  const generation = loads.current;
  L.cancelLiveWork(loads, messages, [reply, summary, director]);
  assert.ok(signals.every((s) => s.aborted));
  assert.equal(reply.current, null);
  assert.equal(summary.current, null);
  // The reply continuation checks both before it summarises or directs.
  assert.notEqual(turn, messages.current);
  assert.notEqual(generation, loads.current);

  const { readFileSync } = await import("node:fs");
  const src = readFileSync(
    new URL(
      "../src/features/flows/player/world/world-player.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    src,
    /useEffect\(\s*\(\) => \(\) =>\s*cancelLiveWork\(loads, messages, \[replyRun, summaryRun, directorRun\]\),\s*\[\],\s*\)/,
  );
});

test("with no chat model a villager answers offline in their own words", () => {
  for (const msg of [
    "No chat model is loaded. Pick a model on this node, or load one first.",
    "Could not load unsloth/Qwen3-4B-Instruct-2507-GGUF: not downloaded",
  ])
    assert.equal(L.isNoChatModelError(msg), true, msg);
  assert.equal(
    L.isNoChatModelError(
      "Could not load org/m: We couldn't connect to 'https://huggingface.co' to load the files",
    ),
    true,
  );
  assert.equal(L.isNoChatModelError("The reply was not valid JSON"), false);
  assert.equal(
    L.isNoChatModelError(
      "Could not load org/m: CUDA out of memory. Tried to allocate 2.00 GiB",
    ),
    false,
  );
  assert.equal(
    L.offlineLine("Welcome to the village!", 0),
    "Welcome to the village!",
  );
  assert.notEqual(
    L.offlineLine("Welcome to the village!", 1),
    "Welcome to the village!",
  );
  assert.ok(L.offlineLine("", 0).length > 0);
});
