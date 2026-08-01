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

const base = {
  originComposer: "item-1",
  currentComposer: "item-1",
  producedTranscript: true,
  baseText: "",
  text: "hello there",
};

test("a transcript submits in the composer the send started in", () => {
  assert.equal(shouldSubmitDictation(base), true);
});

// The composer is reused across thread switches, so a send that lands after
// the move would otherwise submit the destination thread's draft.
test("a thread switch during transcription drops the send", () => {
  assert.equal(
    shouldSubmitDictation({
      ...base,
      currentComposer: "item-2",
      text: "someone else's draft",
    }),
    false,
  );
});

// The identity has to survive a new chat's first persist, which moves
// activeThreadId from null to the remote id without changing the composer.
test("hydrating a new chat keeps the pending send alive", () => {
  assert.equal(shouldSubmitDictation(base), true);
});

test("silence in the original composer sends nothing", () => {
  assert.equal(
    shouldSubmitDictation({
      ...base,
      producedTranscript: false,
      baseText: "draft",
      text: "draft",
    }),
    false,
  );
});

// The plus menu stays open while recording: inserting a saved prompt changes
// the composer without any speech, which text alone cannot tell apart.
test("a menu insertion with no transcript sends nothing", () => {
  assert.equal(
    shouldSubmitDictation({
      ...base,
      producedTranscript: false,
      text: "an inserted saved prompt",
    }),
    false,
  );
});

test("a menu insertion alongside a real transcript still sends", () => {
  assert.equal(
    shouldSubmitDictation({ ...base, text: "an inserted prompt hello there" }),
    true,
  );
});
