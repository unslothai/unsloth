// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The thinking filler stays quiet on short turns and hedges on everything else.

import assert from "node:assert/strict";
import test from "node:test";

import {
  consumePendingFiller,
  fillerDelayMs,
  pickStage2,
  pickWeighted,
  predictFiller,
  setPendingFiller,
  toolKindOf,
} from "../src/features/chat/voice/thinking-filler.ts";

const zero = () => 0;

test("greetings and one-liners get no filler", () => {
  for (const t of ["hello", "hey there", "thanks", "yes", "okay cool", "no"]) {
    assert.equal(predictFiller(t).say, false, t);
  }
});

test("an answer to the assistant's question gets no filler", () => {
  const d = predictFiller("the second one please", { prevAssistantAskedQuestion: true });
  assert.equal(d.say, false);
  assert.equal(
    predictFiller("can you write the second one in python for me please").say,
    true,
  );
});

test("big asks get a hedged sentence, never a promise", () => {
  const d = predictFiller("write me a snake game in python with a score counter", {
    rand: () => 0.9,
  });
  assert.equal(d.say, true);
  assert.equal(d.cls, "task");
  assert.doesNotMatch(d.phrase, /\bI can (totally )?do that\b/i);
});

test("errors route to the fix class", () => {
  const d = predictFiller("my install keeps failing with a weird error", { rand: () => 0.9 });
  assert.equal(d.cls, "fix");
});

test("weighted pick honours the weights", () => {
  const opts = [["a", 1], ["b", 3]] as const;
  assert.equal(pickWeighted(opts, () => 0.1), "a");
  assert.equal(pickWeighted(opts, () => 0.5), "b");
});

test("stage 2 follows the last tool kind, else elapsed", () => {
  assert.equal(pickStage2(["search"], zero), "Looking that up.");
  assert.equal(pickStage2([], zero), "Still on it.");
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
