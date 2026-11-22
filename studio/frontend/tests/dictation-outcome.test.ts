// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  beginDictationSession,
  dictationFailed,
  dictationProducedTranscript,
  markDictationFailed,
  markDictationTranscript,
} from "../src/features/chat/adapters/dictation-outcome.ts";

test("a fresh session has produced nothing and failed at nothing", () => {
  beginDictationSession();
  assert.equal(dictationProducedTranscript(), false);
  assert.equal(dictationFailed(), false);
});

test("a published transcript is visible after the session ends", () => {
  beginDictationSession();
  markDictationTranscript();
  assert.equal(dictationProducedTranscript(), true);
});

test("a reported failure is visible after the session ends", () => {
  beginDictationSession();
  markDictationFailed();
  assert.equal(dictationFailed(), true);
});

test("a partial transcript reports both", () => {
  beginDictationSession();
  markDictationTranscript();
  markDictationFailed();
  assert.equal(dictationProducedTranscript(), true);
  assert.equal(dictationFailed(), true);
});

test("both survive until the next session starts", () => {
  beginDictationSession();
  markDictationTranscript();
  markDictationFailed();
  beginDictationSession();
  assert.equal(dictationProducedTranscript(), false);
  assert.equal(dictationFailed(), false);
});

test("repeated marks in one session stay set", () => {
  beginDictationSession();
  markDictationTranscript();
  markDictationTranscript();
  markDictationFailed();
  markDictationFailed();
  assert.equal(dictationProducedTranscript(), true);
  assert.equal(dictationFailed(), true);
});
