// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  dictationProducedText,
  shouldSubmitDictation,
} from "../src/features/chat/utils/dictation-send.ts";

test("a transcript into an empty composer sends", () => {
  assert.equal(dictationProducedText("", "hello there"), true);
});

test("a transcript appended to a draft sends", () => {
  assert.equal(dictationProducedText("draft", "draft hello there"), true);
});

test("silence with an empty composer sends nothing", () => {
  assert.equal(dictationProducedText("", ""), false);
});

test("silence keeps a pre-recording draft instead of sending it", () => {
  assert.equal(dictationProducedText("draft", "draft"), false);
});

// Anchored on the text at session start, so a final result identical to the
// interim the browser engine already streamed in still counts as produced.
test("a final transcript matching its interim still sends", () => {
  assert.equal(dictationProducedText("", "hello there"), true);
  assert.equal(dictationProducedText("draft", "draft hello there"), true);
});

test("a whitespace-only transcript does not count as text", () => {
  assert.equal(dictationProducedText("draft", "draft  "), false);
  assert.equal(dictationProducedText("", "   "), false);
});

test("a transcript submits in the composer the send started in", () => {
  assert.equal(
    shouldSubmitDictation("t1:t1", "t1:t1", "", "hello there"),
    true,
  );
});

// The composer is reused across thread switches, so a send that lands after
// the move would otherwise submit the destination thread's draft.
test("a thread switch during transcription drops the send", () => {
  assert.equal(
    shouldSubmitDictation("t1:t1", "t2:t2", "", "someone else's draft"),
    false,
  );
});

test("a thread switch drops the send even with a real transcript", () => {
  assert.equal(
    shouldSubmitDictation("t1:t1", "t2:t2", "", "hello there"),
    false,
  );
});

test("silence in the original composer still sends nothing", () => {
  assert.equal(
    shouldSubmitDictation("t1:t1", "t1:t1", "draft", "draft"),
    false,
  );
});
