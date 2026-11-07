// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A pause after "just like" is a pause, not the end of the turn.

import assert from "node:assert/strict";
import test from "node:test";

import { isBackchannel, joinTurn, looksUnfinished } from "../src/features/chat/voice/turn-complete.ts";

test("a sentence cut on a filler or connector is unfinished", () => {
  for (const t of [
    "Could you tell me just like",
    "I was wondering if",
    "So basically the",
    "Tell me about",
    "You know,",
    "And then...",
    "What do you think about, um",
    "It's kind of",
  ]) {
    assert.equal(looksUnfinished(t), true, t);
  }
});

test("a complete thought sends, short ones included", () => {
  for (const t of [
    "Yeah",
    "What is the capital of France?",
    "Tell me a joke.",
    "Stop",
    "Give me three colors",
    "Name the planets in order",
    "Could you tell me just like the three main things about it",
  ]) {
    assert.equal(looksUnfinished(t), false, t);
  }
});

test("terminal punctuation wins over a dangling last word", () => {
  assert.equal(looksUnfinished("Is that what you like?"), false);
  assert.equal(looksUnfinished("I think so."), false);
});

test("joining a held fragment reads as one sentence", () => {
  assert.equal(
    joinTurn("Could you tell me just like,", "The three main things about it"),
    "Could you tell me just like the three main things about it",
  );
  assert.equal(joinTurn("", "Hello"), "Hello");
  assert.equal(joinTurn("And then...", "I left"), "And then I left");
});

test("a hum or breath written as a word is not a turn", () => {
  for (const t of ["Mm.", "Hmm", "mm-hmm", "Uh, um.", "Mmmm", "Ahh"]) assert.equal(isBackchannel(t), true, t);
  for (const t of ["Yeah", "No", "Mm, the second one", "Stop", "Hmm, maybe later"]) assert.equal(isBackchannel(t), false, t);
});
