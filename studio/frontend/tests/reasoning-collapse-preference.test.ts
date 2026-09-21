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
  resolveReasoningToggle,
  startsNewReasoningRound,
} = await import("../src/features/chat/utils/reasoning-visibility.ts");

interface BlockState {
  isStreaming: boolean;
  visibility: DisplayVisibility;
  override: boolean | null;
}

// Mirrors the component: toggle results feed straight back into the open state.
function applyToggle(state: BlockState, open: boolean): BlockState {
  return { ...state, override: resolveReasoningToggle(open, state).override };
}

// Mirrors the component's render-time reset when the setting moves.
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
  // A hand-opened block still wins, including while it is streaming.
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
  // The case the old pair of switches could not express.
  assert.equal(resolveReasoningOpen({ ...base, isStreaming: false }), true);
  // And it is still closable by hand.
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
  // Still closed as the stream continues.
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
  // Collapsed, opened by hand while the model is thinking.
  let state: BlockState = {
    isStreaming: true,
    visibility: "collapsed",
    override: null,
  };
  state = applyToggle(state, true);
  assert.equal(resolveReasoningOpen(state), true);

  // Switched to always expanded mid stream: the block follows the new setting.
  state = applyPreferenceChange(state, "expanded");
  assert.equal(state.override, null);
  assert.equal(resolveReasoningOpen(state), true);

  // Still closable afterwards, which a pinned override would have blocked.
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
  // Still the same round, so a hand-opened block stays open.
  assert.equal(startsNewReasoningRound(true, true), false);
  assert.equal(startsNewReasoningRound(false, true), false);
  assert.equal(startsNewReasoningRound(false, false), false);
});

test("regenerating drops the previous round's override", () => {
  // Block opened by hand after the last round finished.
  let state: BlockState = {
    isStreaming: false,
    visibility: "collapsed",
    override: true,
  };
  assert.equal(resolveReasoningOpen(state), true);

  // Regenerate restarts streaming on the same component instance.
  const wasStreaming = state.isStreaming;
  state = { ...state, isStreaming: true };
  assert.equal(startsNewReasoningRound(state.isStreaming, wasStreaming), true);
  state = { ...state, override: null };
  assert.equal(resolveReasoningOpen(state), false);
});

test("streaming height cap is released only for a block opened against the setting", () => {
  assert.equal(
    resolveReasoningToggle(true, { isStreaming: true, visibility: "collapsed" })
      .releaseStreamingHeight,
    true,
  );
  // Re-opening a block that opened itself keeps the cap, so live text stays scrolled.
  assert.equal(
    resolveReasoningToggle(true, { isStreaming: true, visibility: "auto" })
      .releaseStreamingHeight,
    false,
  );
  assert.equal(
    resolveReasoningToggle(true, { isStreaming: true, visibility: "expanded" })
      .releaseStreamingHeight,
    false,
  );
  // Opening a finished block that auto would have left closed still needs its full height.
  assert.equal(
    resolveReasoningToggle(true, { isStreaming: false, visibility: "auto" })
      .releaseStreamingHeight,
    true,
  );
  assert.equal(
    resolveReasoningToggle(false, { isStreaming: false, visibility: "auto" })
      .releaseStreamingHeight,
    false,
  );
});
