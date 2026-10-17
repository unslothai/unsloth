// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stage-2 fillers and the pending slot the run reads once.

import assert from "node:assert/strict";
import test from "node:test";

import {
  consumePendingFiller,
  fillerDelayMs,
  pickStage2,
  pickWeighted,
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
