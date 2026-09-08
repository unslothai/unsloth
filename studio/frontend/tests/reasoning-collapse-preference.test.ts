// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  reasoningAutoOpensWhileStreaming,
  resolveReasoningOpen,
  resolveReasoningToggle,
  startsNewReasoningRound,
} from "../src/features/chat/utils/reasoning-visibility.ts";

// Mirrors the component: toggle results feed straight back into the open state.
function applyToggle(
  state: {
    isStreaming: boolean;
    collapseByDefault: boolean;
    dismissedWhileStreaming: boolean;
    manualOpen: boolean;
  },
  open: boolean,
) {
  const next = resolveReasoningToggle(open, state);
  return {
    ...state,
    manualOpen: next.manualOpen,
    dismissedWhileStreaming:
      next.dismissedWhileStreaming ?? state.dismissedWhileStreaming,
  };
}

test("thinking streams open and auto-collapses when the preference is off", () => {
  const base = { collapseByDefault: false, manualOpen: false };
  assert.equal(
    resolveReasoningOpen({
      ...base,
      isStreaming: true,
      dismissedWhileStreaming: false,
    }),
    true,
  );
  // Closing mid-stream keeps it closed for the rest of the round.
  assert.equal(
    resolveReasoningOpen({
      ...base,
      isStreaming: true,
      dismissedWhileStreaming: true,
    }),
    false,
  );
  assert.equal(
    resolveReasoningOpen({
      ...base,
      isStreaming: false,
      dismissedWhileStreaming: false,
    }),
    false,
  );
});

test("thinking stays collapsed in both phases when the preference is on", () => {
  const base = { collapseByDefault: true, dismissedWhileStreaming: false };
  assert.equal(
    resolveReasoningOpen({ ...base, isStreaming: true, manualOpen: false }),
    false,
  );
  assert.equal(
    resolveReasoningOpen({ ...base, isStreaming: false, manualOpen: false }),
    false,
  );
  // A hand-opened block still wins, including while it is streaming.
  assert.equal(
    resolveReasoningOpen({ ...base, isStreaming: true, manualOpen: true }),
    true,
  );
  assert.equal(
    resolveReasoningOpen({ ...base, isStreaming: false, manualOpen: true }),
    true,
  );
});

test("auto-open applies only while streaming with the preference off", () => {
  assert.equal(reasoningAutoOpensWhileStreaming(true, false), true);
  assert.equal(reasoningAutoOpensWhileStreaming(true, true), false);
  assert.equal(reasoningAutoOpensWhileStreaming(false, false), false);
  assert.equal(reasoningAutoOpensWhileStreaming(false, true), false);
});

test("a hand-opened block closes again in every phase", () => {
  for (const isStreaming of [true, false]) {
    for (const collapseByDefault of [true, false]) {
      let state = {
        isStreaming,
        collapseByDefault,
        dismissedWhileStreaming: false,
        manualOpen: false,
      };
      state = applyToggle(state, true);
      assert.equal(
        resolveReasoningOpen(state),
        true,
        `open failed for streaming=${isStreaming} collapse=${collapseByDefault}`,
      );
      state = applyToggle(state, false);
      assert.equal(
        resolveReasoningOpen(state),
        false,
        `close failed for streaming=${isStreaming} collapse=${collapseByDefault}`,
      );
    }
  }
});

test("closing still works after the preference flips mid stream", () => {
  // Collapsed by default, opened by hand while the model is thinking.
  let state = {
    isStreaming: true,
    collapseByDefault: true,
    dismissedWhileStreaming: false,
    manualOpen: false,
  };
  state = applyToggle(state, true);
  assert.equal(resolveReasoningOpen(state), true);

  // Preference turned off from settings without leaving the stream.
  state = { ...state, collapseByDefault: false };
  assert.equal(resolveReasoningOpen(state), true);

  // The sticky open has to clear here, or the block cannot be collapsed again.
  state = applyToggle(state, false);
  assert.equal(state.manualOpen, false);
  assert.equal(resolveReasoningOpen(state), false);
});

test("a round starts only when streaming resumes", () => {
  assert.equal(startsNewReasoningRound(true, false), true);
  // Still the same round, so a hand-opened block stays open.
  assert.equal(startsNewReasoningRound(true, true), false);
  assert.equal(startsNewReasoningRound(false, true), false);
  assert.equal(startsNewReasoningRound(false, false), false);
});

test("regenerating drops the previous round's manual open", () => {
  // Block opened by hand after the last round finished.
  let state = {
    isStreaming: false,
    collapseByDefault: true,
    dismissedWhileStreaming: false,
    manualOpen: true,
  };
  assert.equal(resolveReasoningOpen(state), true);

  // Regenerate restarts streaming on the same component instance.
  const wasStreaming = state.isStreaming;
  state = { ...state, isStreaming: true };
  assert.equal(startsNewReasoningRound(state.isStreaming, wasStreaming), true);
  state = { ...state, manualOpen: false, dismissedWhileStreaming: false };
  assert.equal(resolveReasoningOpen(state), false);
});

test("streaming height cap is released only for a hand-opened block", () => {
  assert.equal(
    resolveReasoningToggle(true, { isStreaming: true, collapseByDefault: true })
      .releaseStreamingHeight,
    true,
  );
  // Re-opening an auto-opened block keeps the cap, so live text stays scrolled.
  assert.equal(
    resolveReasoningToggle(true, {
      isStreaming: true,
      collapseByDefault: false,
    }).releaseStreamingHeight,
    false,
  );
  assert.equal(
    resolveReasoningToggle(false, {
      isStreaming: false,
      collapseByDefault: false,
    }).releaseStreamingHeight,
    false,
  );
});
