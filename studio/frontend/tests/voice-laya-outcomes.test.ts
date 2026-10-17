// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The facts the voice loop reports to the decision log: a repeated utterance, a reply
// that echoed its filler, and which interrupt verdict each one contradicts.

import assert from "node:assert/strict";
import test from "node:test";

import {
  interjectOutcomeFor,
  REPEAT_WINDOW_MS,
  replyEchoesFiller,
  sharesMostWords,
} from "../src/features/chat/voice/laya-outcomes.ts";

test("sharesMostWords ignores stop words and needs most content words back", () => {
  assert.equal(sharesMostWords("can you make it shorter", "please make it shorter"), true);
  assert.equal(sharesMostWords("make it shorter", "what about the weather"), false);
  assert.equal(sharesMostWords("um okay", "okay sure"), false);
  assert.equal(sharesMostWords("", "anything"), false);
});

test("an ignored utterance said again inside the window is repeated, a queued one reasked", () => {
  assert.equal(interjectOutcomeFor("ignore", "make it shorter", "make it shorter please", 3000), "repeated");
  assert.equal(interjectOutcomeFor("queue", "set a timer for ten minutes", "timer for ten minutes", 9000), "reasked");
  assert.equal(interjectOutcomeFor("ignore", "make it shorter", "make it shorter", REPEAT_WINDOW_MS + 1), "ok");
  assert.equal(interjectOutcomeFor("ignore", "make it shorter", "tell me a joke", 2000), "ok");
  assert.equal(interjectOutcomeFor("ignore", "make it shorter", null, 2000), "ok");
  assert.equal(interjectOutcomeFor("restart", "make it shorter", "make it shorter", 100), "ok");
});

test("replyEchoesFiller spots the filler phrase at the head of the reply", () => {
  assert.equal(replyEchoesFiller("Let me look into the weather. The forecast is", "Let me look into the weather."), true);
  assert.equal(replyEchoesFiller("Sure, let me look into the weather!", "let me look into the weather"), true);
  assert.equal(replyEchoesFiller("The forecast is sunny all week.", "Let me look into the weather."), false);
  assert.equal(replyEchoesFiller("Anything", ""), false);
});
