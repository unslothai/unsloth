// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createHearingHint } from "../src/features/chat/voice/hearing-hint.ts";

test("from thinking, transcribing or generating, the first loud frame lights hearing", () => {
  for (const orb of [
    "thinking",
    "transcribing",
    "generating",
    "synthesizing",
  ]) {
    const h = createHearingHint();
    assert.equal(h.step(false, orb, 0), null);
    assert.equal(h.step(true, orb, 10), true);
    assert.equal(h.step(true, "hearing", 20), null);
  }
});

test("while the reply is audible, loud frames leave the orb alone (the barge gate decides)", () => {
  const h = createHearingHint();
  assert.equal(h.step(true, "speaking", 0), null);
  assert.equal(h.on, false);
});

test("a gap between syllables keeps it; a real pause lets it go", () => {
  const h = createHearingHint(500);
  h.step(true, "thinking", 0);
  assert.equal(h.step(false, "hearing", 300), null);
  assert.equal(h.step(true, "hearing", 400), null);
  assert.equal(h.step(false, "hearing", 800), null);
  assert.equal(h.step(false, "hearing", 950), false);
});

test("the detector saying no voice drops it at once; a committed barge-in keeps it", () => {
  const h = createHearingHint();
  h.step(true, "generating", 0);
  assert.equal(h.reject(), false);
  assert.equal(h.reject(), null);
  h.step(true, "generating", 10);
  h.commit();
  assert.equal(h.step(false, "hearing", 5000), null);
});
