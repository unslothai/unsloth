// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A give-up is not always how the turn ended: a tool run sends the notice and then BREAKS
 * INTO the final answering pass, which usually stops normally, so a completed answer was
 * stamped `paused`. `length` must keep the old behaviour, being the shape a give-up ends on.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  completedAfterGivingUp,
  type IncompleteReason,
} from "../src/features/chat/utils/continuation.ts";

test("a finished answer clears the give-up, a length stop does not", () => {
  assert.equal(completedAfterGivingUp("stop"), true);
  assert.equal(completedAfterGivingUp("tool_calls"), true);
  assert.equal(
    completedAfterGivingUp("length"),
    false,
    "a give-up ends on length, so reading it as success would erase the real case",
  );
  assert.equal(completedAfterGivingUp(null), false);
  assert.equal(completedAfterGivingUp(undefined), false);
  assert.equal(completedAfterGivingUp(""), false);
});

/** The adapter's own two steps over a stream, in the order it runs them. */
function replay(
  chunks: readonly { finishReason?: string; gaveUp?: boolean }[],
): IncompleteReason | null {
  let incompleteReason: IncompleteReason | null = null;
  let preemptGaveUp = false;
  for (const chunk of chunks) {
    if (chunk.gaveUp) {
      preemptGaveUp = true;
      continue;
    }
    if (chunk.finishReason === "length") {
      incompleteReason = "length";
    } else if (chunk.finishReason) {
      incompleteReason = null;
      if (completedAfterGivingUp(chunk.finishReason)) {
        preemptGaveUp = false;
      }
    }
  }
  if (preemptGaveUp) {
    incompleteReason = "paused";
  }
  return incompleteReason;
}

test("a turn that gave up mid-round and then answered is complete", () => {
  assert.equal(
    replay([
      { finishReason: "tool_calls" },
      { gaveUp: true },
      // The final synthesis pass the backend breaks into, finishing normally.
      { finishReason: "stop" },
    ]),
    null,
    "a completed answer was labelled paused, kept the 'did not get it back' notice and " +
      "offered a Continue with nothing to continue",
  );
});

test("a turn that gave up and stopped there is still paused", () => {
  assert.equal(
    replay([{ gaveUp: true }, { finishReason: "length" }]),
    "paused",
    "this is the case the override exists for: the backend could not get the model " +
      "back, and `paused` is also what refuses the automatic continuation",
  );
});

test("a give-up after an answer still wins", () => {
  // The order that matters: the notice arrives AFTER a pass that stopped normally when the
  // give-up happened in the final pass itself, which really did end the turn early.
  assert.equal(replay([{ finishReason: "stop" }, { gaveUp: true }]), "paused");
});

test("nothing else about the length latch changes", () => {
  assert.equal(replay([{ finishReason: "length" }]), "length");
  assert.equal(replay([{ finishReason: "length" }, { finishReason: "stop" }]), null);
  assert.equal(replay([]), null);
});

test("the adapter clears the latch where it latches the finish reason", () => {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  const latch = source.indexOf('if (chunk.choices?.[0]?.finish_reason === "length")');
  assert.ok(latch > 0, "the finish-reason latch moved; this test has to move with it");
  const block = source.slice(latch, latch + 1400);
  assert.match(
    block,
    /completedAfterGivingUp\(chunk\.choices\[0\]\.finish_reason\)/,
    "the give-up latch is never cleared, so a completed answer is still stamped paused",
  );
  assert.match(block, /preemptGaveUp = false;/);
  assert.ok(
    source.includes("if (preemptGaveUp) {"),
    "the override itself must stay: a turn that really gave up is paused",
  );
});
