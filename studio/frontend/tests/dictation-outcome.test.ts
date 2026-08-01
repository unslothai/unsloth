// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  dictationFailed,
  markDictationFailed,
  resetDictationFailure,
} from "../src/features/chat/adapters/dictation-outcome.ts";

test("a fresh session reports no failure", () => {
  resetDictationFailure();
  assert.equal(dictationFailed(), false);
});

test("a reported failure is visible after the session ends", () => {
  resetDictationFailure();
  markDictationFailed();
  assert.equal(dictationFailed(), true);
});

// The recording bar reads the flag once the session is gone, so it has to
// survive until the next session starts rather than clearing on end.
test("the failure survives until the next session starts", () => {
  resetDictationFailure();
  markDictationFailed();
  assert.equal(dictationFailed(), true);
  resetDictationFailure();
  assert.equal(dictationFailed(), false);
});

test("repeated failures in one session stay set", () => {
  resetDictationFailure();
  markDictationFailed();
  markDictationFailed();
  assert.equal(dictationFailed(), true);
});
