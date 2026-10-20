// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A pause after "just like" is a pause, not the end of the turn.

import assert from "node:assert/strict";
import test from "node:test";

import { isBackchannel, isSttNoise, joinTurn } from "../src/features/chat/voice/turn-complete.ts";

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

test("the phrase STT writes for a cough is not a turn", () => {
  for (const t of ["Thank you.", "You", "thanks for watching", "Bye."]) assert.equal(isSttNoise(t), true, t);
  for (const t of ["Thank you, that helps", "You there?", "No thanks", "Bye for now"]) assert.equal(isSttNoise(t), false, t);
});
