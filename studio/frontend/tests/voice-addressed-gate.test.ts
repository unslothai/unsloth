// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The pure half of the addressed gate: the name rule, the overheard buffer and the
// context block it quotes into the next turn, and the outcome rule.

import assert from "node:assert/strict";
import test from "node:test";

import {
  addressedOutcome,
  disownsReply,
  isAddressedByName,
  OVERHEARD_MAX_CHARS,
  OVERHEARD_WINDOW_MS,
  overheardText,
  pushOverheard,
  withOverheardContext,
} from "../src/features/chat/voice/addressed-gate.ts";
import { REPEAT_WINDOW_MS } from "../src/features/chat/voice/laya-outcomes.ts";

test("naming the app forces the turn to the assistant", () => {
  assert.equal(isAddressedByName("hey assistant, what's the weather"), true);
  assert.equal(isAddressedByName("okay unsloth read that back"), true);
  assert.equal(isAddressedByName("Assistant: summarize this"), true);
  assert.equal(isAddressedByName("my computer is slow today"), false);
  assert.equal(isAddressedByName("grandma, say something to it"), false);
  assert.equal(isAddressedByName("watch, it can do this"), false);
});

test("the overheard buffer keeps the window and the char budget", () => {
  const t0 = 1_000_000;
  let list = pushOverheard([], { text: "pass the salt", label: "other", at: t0 });
  list = pushOverheard(list, { text: "no the other one", label: "other", at: t0 + 1000 });
  assert.equal(overheardText(list, t0 + 2000), "pass the salt no the other one");
  const later = pushOverheard(list, { text: "fresh", label: "reading", at: t0 + OVERHEARD_WINDOW_MS + 1 });
  assert.deepEqual(later.map((e) => e.text), ["no the other one", "fresh"]);
  assert.equal(overheardText(later, t0 + OVERHEARD_WINDOW_MS + 1), "no the other one fresh");
  assert.equal(overheardText(later, t0 + 2 * OVERHEARD_WINDOW_MS + 2), "");
  const long = Array.from({ length: 20 }, (_, i) => ({ text: `sentence number ${i} is fairly long indeed`, label: "other" as const, at: t0 }));
  const text = overheardText(long, t0);
  assert.ok(text.length <= OVERHEARD_MAX_CHARS);
  assert.ok(text.endsWith("number 19 is fairly long indeed"));
});

test("the next turn carries the overheard speech only when there is some", () => {
  const t0 = 5_000;
  assert.equal(withOverheardContext("what time is it", [], t0), "what time is it");
  const list = [{ text: 'he said "later"', label: "other" as const, at: t0 }];
  assert.equal(
    withOverheardContext("what time is it", list, t0 + 10),
    "(Heard earlier, not addressed to you: \"he said 'later'\")\n\nwhat time is it",
  );
});

test("a held-back turn said again inside the window was for the assistant", () => {
  assert.equal(addressedOutcome("set a timer for ten minutes", "set a timer for ten minutes please", 3000), "repeated");
  assert.equal(addressedOutcome("set a timer for ten minutes", "tell me a joke", 3000), "ok");
  assert.equal(addressedOutcome("set a timer", "set a timer", REPEAT_WINDOW_MS + 1), "ok");
  assert.equal(addressedOutcome("set a timer", null, 100), "ok");
  assert.equal(disownsReply("no not you, I was talking to Sam"), true);
  assert.equal(disownsReply("I wasn't talking to you"), true);
  assert.equal(disownsReply("thanks that's great"), false);
});
