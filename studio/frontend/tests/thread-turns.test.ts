// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  threadTurns,
  turnNumberAt,
  turnOpenerIdAt,
} from "../src/components/assistant-ui/thread-turns.ts";

const thread = [
  { id: "greeting", role: "assistant" },
  { id: "u1", role: "user" },
  { id: "a1", role: "assistant" },
  { id: "u2", role: "user" },
  { id: "u3", role: "user" },
  { id: "a3", role: "assistant" },
  { id: "a3-more", role: "assistant" },
];

test("each user message opens a turn and its replies share it", () => {
  assert.deepEqual(
    thread.map((_, index) => turnNumberAt(thread, index)),
    [0, 1, 1, 2, 3, 3, 3],
  );
  assert.deepEqual(threadTurns(thread).openerIds, ["u1", "u2", "u3"]);
});

test("a reply resolves to the user message that opened its turn", () => {
  assert.equal(turnOpenerIdAt(thread, 0), undefined);
  assert.equal(turnOpenerIdAt(thread, 1), "u1");
  assert.equal(turnOpenerIdAt(thread, 2), "u1");
  assert.equal(turnOpenerIdAt(thread, 6), "u3");
});

test("one pass per message array, and a streamed reply keeps the signature", () => {
  assert.equal(threadTurns(thread), threadTurns(thread));
  // a streamed token rebuilds the array without adding a turn
  const streamed = [...thread];
  assert.notEqual(threadTurns(streamed), threadTurns(thread));
  assert.equal(threadTurns(streamed).signature, threadTurns(thread).signature);
  // a branch switch at turn 2 replaces the later user messages
  const switched = [...thread.slice(0, 3), { id: "u2-edit", role: "user" }];
  assert.notEqual(
    threadTurns(switched).signature,
    threadTurns(thread).signature,
  );
});
