// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  dictationProducedText,
  dictationSendBlocked,
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

const open = {
  composerDisabled: false,
  uploading: false,
  researchActive: false,
  runActive: false,
  queueDisabled: false,
  hasOverlay: false,
  hasAttachments: false,
  hasPendingAudio: false,
};

test("an idle composer accepts the dictation send", () => {
  assert.equal(dictationSendBlocked(open), false);
});

test("the composer's own unavailable states block regardless of a run", () => {
  assert.equal(dictationSendBlocked({ ...open, composerDisabled: true }), true);
  assert.equal(dictationSendBlocked({ ...open, uploading: true }), true);
  assert.equal(dictationSendBlocked({ ...open, researchActive: true }), true);
});

test("a running response alone does not block", () => {
  assert.equal(dictationSendBlocked({ ...open, runActive: true }), false);
});

test("non-queueable content blocks only while a run is active", () => {
  for (const key of [
    "queueDisabled",
    "hasOverlay",
    "hasAttachments",
    "hasPendingAudio",
  ] as const) {
    assert.equal(dictationSendBlocked({ ...open, [key]: true }), false);
    assert.equal(
      dictationSendBlocked({ ...open, runActive: true, [key]: true }),
      true,
    );
  }
});

test("an attachment completing mid-transcription blocks a queued send", () => {
  assert.equal(
    dictationSendBlocked({
      ...open,
      runActive: true,
      uploading: false,
      hasAttachments: true,
    }),
    true,
  );
});
