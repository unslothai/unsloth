// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  allowsPassiveAgentScroll,
  completeAgentHistoryReveal,
  initialAgentHistoryNavigation,
  interruptAgentHistoryReveal,
  markAgentHistoryRevealPositioned,
  requestAgentHistoryReveal,
} from "../src/components/assistant-ui/agent-history-navigation.ts";

test("search owns the navigation until its mounted row has been positioned and verified", () => {
  const reader = initialAgentHistoryNavigation();
  const requested = requestAgentHistoryReveal(reader, 5, 400);
  assert.deepEqual(requested, { kind: "search", epoch: 1, row: 5, positioned: false });
  assert.equal(allowsPassiveAgentScroll(requested), false);
  assert.deepEqual(completeAgentHistoryReveal(requested, 1, true), requested);
  const positioned = markAgentHistoryRevealPositioned(requested, 1);
  assert.equal(allowsPassiveAgentScroll(positioned), false);
  assert.deepEqual(completeAgentHistoryReveal(positioned, 1, false), positioned);
  assert.deepEqual(completeAgentHistoryReveal(positioned, 1, true), { kind: "reader", epoch: 1 });
});

test("a newer search invalidates stale layout/animation completions", () => {
  const first = requestAgentHistoryReveal(initialAgentHistoryNavigation(), 5, 400);
  const second = requestAgentHistoryReveal(first, 399, 400);
  assert.deepEqual(markAgentHistoryRevealPositioned(second, first.epoch), second);
  assert.deepEqual(completeAgentHistoryReveal(second, first.epoch, true), second);
  assert.deepEqual(second, { kind: "search", epoch: 2, row: 399, positioned: false });
  const atLast = markAgentHistoryRevealPositioned(second, second.epoch);
  assert.deepEqual(completeAgentHistoryReveal(atLast, second.epoch, true), { kind: "reader", epoch: 2 });
});

test("repeating a one-result search requires a new navigation commit even on the same row", () => {
  const first = requestAgentHistoryReveal(initialAgentHistoryNavigation(), 5, 400);
  const firstDone = completeAgentHistoryReveal(markAgentHistoryRevealPositioned(first, first.epoch), first.epoch, true);
  const repeated = requestAgentHistoryReveal(firstDone, 5, 400);
  assert.deepEqual(repeated, { kind: "search", epoch: 2, row: 5, positioned: false });
  assert.equal(allowsPassiveAgentScroll(repeated), false);
  assert.deepEqual(completeAgentHistoryReveal(repeated, first.epoch, true), repeated);
});

test("manual reader gesture cancels search ownership and invalidates in-flight completion", () => {
  const requested = requestAgentHistoryReveal(initialAgentHistoryNavigation(), 200, 400);
  const interrupted = interruptAgentHistoryReveal(requested);
  assert.deepEqual(interrupted, { kind: "reader", epoch: 2 });
  assert.equal(allowsPassiveAgentScroll(interrupted), true);
  assert.deepEqual(completeAgentHistoryReveal(interrupted, requested.epoch, true), interrupted);
  assert.deepEqual(interruptAgentHistoryReveal(interrupted), interrupted);
});

test("invalid search targets do not mutate navigation state", () => {
  const previous = initialAgentHistoryNavigation();
  for (const row of [-1, 400, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.deepEqual(requestAgentHistoryReveal(previous, row, 400), previous);
  }
  assert.deepEqual(requestAgentHistoryReveal(previous, 0, 0), previous);
});
