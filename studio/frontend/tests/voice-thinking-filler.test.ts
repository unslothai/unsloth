// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stage-2 fillers and the pending slot the run reads once.

import assert from "node:assert/strict";
import test from "node:test";

import {
  consumePendingFiller,
  fillerDelayMs,
  pickStage2,
  pickApproval,
  asidePhrases,
  consumePendingReplace,
  replaceSystemNote,
  setPendingReplace,
  pickQuietFallback,
  pickWriting,
  pickWeighted,
  fillerStandIns,
  setPendingFiller,
  toolKindOf,
} from "../src/features/chat/voice/thinking-filler.ts";

const zero = () => 0;

test("weighted pick honours the weights", () => {
  const opts = [["a", 1], ["b", 3]] as const;
  assert.equal(pickWeighted(opts, () => 0.1), "a");
  assert.equal(pickWeighted(opts, () => 0.5), "b");
});

test("stage 2 follows the last tool kind, else elapsed", () => {
  assert.equal(pickStage2(["search"], zero), "Looking that up.");
  assert.equal(pickStage2([], zero), "Still thinking.");
  assert.equal(pickStage2([], zero, true), "Let me think.");
  assert.equal(pickStage2(["search"], zero, true), "Looking that up.");
  assert.equal(toolKindOf("web_search"), "search");
  assert.equal(toolKindOf("read_skill"), "file");
  assert.equal(toolKindOf("run_python"), "code");
});

test("delay is never instant and the pending slot reads once", () => {
  assert.ok(fillerDelayMs(zero) >= 300);
  setPendingFiller({ say: true, cls: "task", phrase: "Let me see." });
  assert.equal(consumePendingFiller()?.phrase, "Let me see.");
  assert.equal(consumePendingFiller(), null);
  setPendingFiller({ say: false, cls: "none", phrase: "" });
  assert.equal(consumePendingFiller(), null);
});

test("the quiet fallback and the writing aside pick from their own banks", () => {
  const zero = () => 0;
  assert.equal(pickQuietFallback(zero), "Hmm.");
  assert.equal(pickWriting(zero), "One moment.");
});

test("a replace interrupt is handed to the next run once", () => {
  assert.equal(consumePendingReplace(), false);
  setPendingReplace(true);
  assert.equal(consumePendingReplace(), true);
  assert.equal(consumePendingReplace(), false);
  assert.match(replaceSystemNote(), /latest message/);
});

test("the approval line names the Allow card and is warmed with the other asides", () => {
  assert.match(pickApproval(() => 0), /Allow/);
  for (const p of [pickApproval(() => 0), pickApproval(() => 0.99)]) assert.ok(asidePhrases().includes(p));
});

// The model decides the answer, so nothing said before it may sound like a yes, a no or a promise.
const AGREEMENT_OR_REFUSAL =
  /\b(ok|okay|sure|got it|alright|all right|yes|yeah|yep|no|nope|no problem|sounds good|will do|on it|of course|absolutely|good question|(?<!if )i can do that|can't|cannot|right|almost there|here we go)\b/i;

test("no aside implies yes, no or an outcome", () => {
  const all = [...asidePhrases(), ...fillerStandIns()];
  assert.ok(all.length > 10);
  for (const p of all) assert.doesNotMatch(p, AGREEMENT_OR_REFUSAL, p);
  for (const p of ["Okay.", "Sure, one sec.", "Got it.", "No problem.", "I'm on it."]) {
    assert.match(p, AGREEMENT_OR_REFUSAL);
  }
});

test("the first aside of a turn never says still", () => {
  for (let i = 0; i < 20; i++) assert.doesNotMatch(pickStage2([], () => i / 20, true), /still/i);
});
