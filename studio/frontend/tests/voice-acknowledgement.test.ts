// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createClipSpeculation } from "../src/features/chat/voice/clip-speculation.ts";
import { isAcknowledgement, isBackchannel } from "../src/features/chat/voice/turn-complete.ts";

test("a word of agreement over a reply is not a turn", () => {
  const telling = "The capital of France is Paris, and it has been since";
  for (const said of ["Okay.", "Right!", "I see.", "Got it", "Makes sense.", "yeah", "Yes."]) {
    assert.equal(isAcknowledgement(said, telling), true, said);
  }
  for (const said of ["Okay, stop.", "Right, but why?", "Stop.", "No.", "What?", "Paris"]) {
    assert.equal(isAcknowledgement(said, telling), false, said);
  }
  // After a question, yes is the answer; "okay" still only acknowledges.
  const asking = "Shall I book the table for eight?";
  assert.equal(isAcknowledgement("Yes.", asking), false);
  assert.equal(isAcknowledgement("Yeah", asking), false);
  assert.equal(isAcknowledgement("Okay.", asking), true);
  assert.equal(isAcknowledgement("mm-hmm", telling), false);
  assert.equal(isBackchannel("mm-hmm"), true);
});
