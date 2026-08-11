// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  reasoningToggleTargetsManualState,
  resolveReasoningOpen,
} from "../src/features/chat/utils/reasoning-visibility.ts";

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

test("mid-stream toggles route to the manual flag only when collapsing by default", () => {
  assert.equal(reasoningToggleTargetsManualState(true, false), false);
  assert.equal(reasoningToggleTargetsManualState(true, true), true);
  assert.equal(reasoningToggleTargetsManualState(false, false), true);
  assert.equal(reasoningToggleTargetsManualState(false, true), true);
});
