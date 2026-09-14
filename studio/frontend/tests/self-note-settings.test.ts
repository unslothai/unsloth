// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_SELF_NOTE_ENABLED,
  DEFAULT_SELF_NOTE_RESERVE_TOKENS,
  sanitizeSelfNoteEnabled,
  sanitizeSelfNoteReserveTokens,
} from "../src/features/chat/utils/auto-compaction.ts";

test("the self-note feature defaults off", () => {
  assert.equal(DEFAULT_SELF_NOTE_ENABLED, false);
});

test("the enabled flag takes a boolean and nothing else", () => {
  assert.equal(sanitizeSelfNoteEnabled(true), true);
  assert.equal(sanitizeSelfNoteEnabled(false), false);
  assert.equal(sanitizeSelfNoteEnabled("true"), undefined);
  assert.equal(sanitizeSelfNoteEnabled(undefined), undefined);
});

test("the reserve clamps into the range the backend accepts", () => {
  // The backend is ge=64 le=4096 and 400s the whole save on one bad field, so
  // the client must never send an out-of-range value.
  assert.equal(sanitizeSelfNoteReserveTokens(8), 64);
  assert.equal(sanitizeSelfNoteReserveTokens(100000), 4096);
  assert.equal(sanitizeSelfNoteReserveTokens(512), 512);
});

test("a fractional reserve rounds to an integer", () => {
  assert.equal(sanitizeSelfNoteReserveTokens(512.7), 513);
});

test("a reserve that is not a finite number is rejected", () => {
  assert.equal(sanitizeSelfNoteReserveTokens("512"), undefined);
  assert.equal(sanitizeSelfNoteReserveTokens(Number.NaN), undefined);
  assert.equal(sanitizeSelfNoteReserveTokens(Number.POSITIVE_INFINITY), undefined);
});

test("the default reserve is inside its own range", () => {
  assert.equal(
    sanitizeSelfNoteReserveTokens(DEFAULT_SELF_NOTE_RESERVE_TOKENS),
    DEFAULT_SELF_NOTE_RESERVE_TOKENS,
  );
});
