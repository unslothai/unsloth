// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

// reasoning-visibility.ts imports a sibling extensionless, like the rest of src.
registerBundlerResolver();

import type { DisplayVisibility } from "../src/features/chat/utils/display-visibility.ts";
const { DISPLAY_VISIBILITIES } = await import(
  "../src/features/chat/utils/display-visibility.ts"
);
const {
  reasoningFollowsPreference,
  resolveReasoningOpen,
  startsNewReasoningRound,
} = await import("../src/features/chat/utils/reasoning-visibility.ts");

interface BlockState {
  isStreaming: boolean;
  visibility: DisplayVisibility;
  override: boolean | null;
}

function applyToggle(state: BlockState, open: boolean): BlockState {
  return { ...state, override: open };
}

function applyPreferenceChange(
  state: BlockState,
  visibility: DisplayVisibility,
): BlockState {
  return { ...state, visibility, override: null };
}

test("auto streams the block open and collapses it when the stream ends", () => {
  const base = { visibility: "auto" as const, override: null };
  assert.equal(resolveReasoningOpen({ ...base, isStreaming: true }), true);
  assert.equal(resolveReasoningOpen({ ...base, isStreaming: false }), false);
});

test("collapsed keeps the block shut in both phases", () => {
  const base = { visibility: "collapsed" as const, override: null };
  assert.equal(resolveReasoningOpen({ ...base, isStreaming: true }), false);
  assert.equal(resolveReasoningOpen({ ...base, isStreaming: false }), false);
  assert.equal(
    resolveReasoningOpen({ ...base, isStreaming: true, override: true }),
    true,
  );
  assert.equal(
    resolveReasoningOpen({ ...base, isStreaming: false, override: true }),
    true,
  );
});

test("always expanded keeps the block open in both phases, streaming included", () => {
  const base = { visibility: "expanded" as const, override: null };
  assert.equal(resolveReasoningOpen({ ...base, isStreaming: true }), true);
  assert.equal(resolveReasoningOpen({ ...base, isStreaming: false }), true);
  assert.equal(
    resolveReasoningOpen({ ...base, isStreaming: false, override: false }),
    false,
  );
});

test("closing mid-stream keeps it closed for the rest of the round", () => {
  let state: BlockState = {
    isStreaming: true,
    visibility: "auto",
    override: null,
  };
  assert.equal(resolveReasoningOpen(state), true);
  state = applyToggle(state, false);
  assert.equal(resolveReasoningOpen(state), false);
  assert.equal(resolveReasoningOpen({ ...state, isStreaming: true }), false);
});

test("a block sits where the setting puts it until someone touches it", () => {
  assert.equal(reasoningFollowsPreference(true, true, "auto"), true);
  assert.equal(reasoningFollowsPreference(false, true, "auto"), false);
  assert.equal(reasoningFollowsPreference(true, false, "expanded"), true);
  assert.equal(reasoningFollowsPreference(false, false, "collapsed"), true);
  assert.equal(reasoningFollowsPreference(true, false, "collapsed"), false);
});

test("a hand-opened block closes again in every phase and every setting", () => {
  for (const isStreaming of [true, false]) {
    for (const visibility of DISPLAY_VISIBILITIES) {
      let state: BlockState = { isStreaming, visibility, override: null };
      state = applyToggle(state, true);
      assert.equal(
        resolveReasoningOpen(state),
        true,
        `open failed for streaming=${isStreaming} visibility=${visibility}`,
      );
      state = applyToggle(state, false);
      assert.equal(
        resolveReasoningOpen(state),
        false,
        `close failed for streaming=${isStreaming} visibility=${visibility}`,
      );
    }
  }
});

test("changing the setting mid stream re-applies it to a block already on screen", () => {
  let state: BlockState = {
    isStreaming: true,
    visibility: "collapsed",
    override: null,
  };
  state = applyToggle(state, true);
  assert.equal(resolveReasoningOpen(state), true);

  state = applyPreferenceChange(state, "expanded");
  assert.equal(state.override, null);
  assert.equal(resolveReasoningOpen(state), true);

  state = applyToggle(state, false);
  assert.equal(resolveReasoningOpen(state), false);
});

test("switching to collapsed shuts a block the user had opened", () => {
  let state: BlockState = {
    isStreaming: false,
    visibility: "expanded",
    override: null,
  };
  state = applyToggle(state, true);
  state = applyPreferenceChange(state, "collapsed");
  assert.equal(resolveReasoningOpen(state), false);
});

test("a round starts only when streaming resumes", () => {
  assert.equal(startsNewReasoningRound(true, false), true);
  assert.equal(startsNewReasoningRound(true, true), false);
  assert.equal(startsNewReasoningRound(false, true), false);
  assert.equal(startsNewReasoningRound(false, false), false);
});

test("regenerating drops the previous round's override", () => {
  let state: BlockState = {
    isStreaming: false,
    visibility: "collapsed",
    override: true,
  };
  assert.equal(resolveReasoningOpen(state), true);

  const wasStreaming = state.isStreaming;
  state = { ...state, isStreaming: true };
  assert.equal(startsNewReasoningRound(state.isStreaming, wasStreaming), true);
  state = { ...state, override: null };
  assert.equal(resolveReasoningOpen(state), false);
});
