// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";
import { agentWindowAtIndex } from "../src/components/assistant-ui/agent-history-scroll-window.ts";
import {
  allowsAgentReaderScrollIntent,
  allowsRecentAgentBoundaryReconcile,
  consumeAgentReaderScrollIntent,
  initialAgentReaderScrollIntent,
  requestAgentReaderScrollIntent,
} from "../src/components/assistant-ui/agent-history-navigation.ts";

test("a layout/anchor scroll without fresh reader input cannot start a window seek", () => {
  const initial = initialAgentReaderScrollIntent();
  assert.equal(allowsAgentReaderScrollIntent(initial, initial.epoch), false);
  const gesture = requestAgentReaderScrollIntent(initial);
  assert.equal(allowsAgentReaderScrollIntent(gesture, gesture.epoch), true);
  const settled = consumeAgentReaderScrollIntent(gesture, gesture.epoch);
  assert.equal(allowsAgentReaderScrollIntent(settled, gesture.epoch), false);
  // 200 redundant native/synthetic scroll events from two oscillating DOM windows
  // must never become 200 new reader-navigation transactions.
  for (let i = 0; i < 200; i++) {
    assert.equal(allowsAgentReaderScrollIntent(settled, gesture.epoch), false);
    assert.deepEqual(consumeAgentReaderScrollIntent(settled, gesture.epoch), settled);
  }
  const second = requestAgentReaderScrollIntent(settled);
  assert.equal(allowsAgentReaderScrollIntent(second, second.epoch), true);
});

test("superseded rAF scroll callbacks cannot consume a newer gesture", () => {
  const first = requestAgentReaderScrollIntent(initialAgentReaderScrollIntent());
  const next = requestAgentReaderScrollIntent(first);
  assert.equal(allowsAgentReaderScrollIntent(next, first.epoch), false);
  assert.deepEqual(consumeAgentReaderScrollIntent(next, first.epoch), next);
  assert.equal(allowsAgentReaderScrollIntent(next, next.epoch), true);
  const consumed = consumeAgentReaderScrollIntent(next, next.epoch);
  assert.equal(consumed.sampledEpoch, next.epoch);
});

test("the production scroll handler gates passive seeks by gesture epoch", () => {
  const source = readSrc("components/assistant-ui/agent-history-scroll-messages.tsx");
  assert.match(source, /allowsAgentReaderScrollIntent\(/);
  assert.match(source, /consumeAgentReaderScrollIntent\(/);
  assert.match(source, /requestAgentReaderScrollIntent\(/);
});

test("the observed A/B oscillation requires fresh intent before every new window", () => {
  const windowA = { start: 99, end: 104 };
  const windowB = { start: 101, end: 106 };
  assert.deepEqual(agentWindowAtIndex(1000, 103, windowA, 5), windowB);
  assert.deepEqual(agentWindowAtIndex(1000, 101, windowB, 5), windowA);

  let intent = requestAgentReaderScrollIntent(initialAgentReaderScrollIntent());
  let window = windowA;
  let transitions = 0;
  for (let i = 0; i < 200; i++) {
    if (!allowsAgentReaderScrollIntent(intent, intent.epoch)) continue;
    const next = agentWindowAtIndex(1000, i % 2 === 0 ? 103 : 101, window, 5);
    if (next.start !== window.start || next.end !== window.end) {
      transitions++;
      window = next;
      intent = consumeAgentReaderScrollIntent(intent, intent.epoch);
    }
  }
  assert.deepEqual(window, windowB);
  assert.equal(transitions, 1);
});

test("a consumed gesture may finish at the physical boundary, never on passive scroll echoes", () => {
  const gesture = requestAgentReaderScrollIntent(initialAgentReaderScrollIntent());
  const consumed = consumeAgentReaderScrollIntent(gesture, gesture.epoch);
  const at = (
    intent: typeof consumed, timestamp: number, now: number,
    start: number, end: number, total: number,
    top: boolean, bottom: boolean,
  ) => allowsRecentAgentBoundaryReconcile(
    intent, timestamp, now, start, end, total, top, bottom,
  );
  assert.equal(at(consumed, 100, 150, 2002, 2007, 4000, false, true), true);
  assert.equal(at(consumed, 100, 150, 2002, 2007, 4000, true, false), true);
  assert.equal(at(consumed, 100, 150, 2002, 4000, 4000, false, true), false);
  assert.equal(at(consumed, 100, 150, 0, 5, 4000, true, false), false);
  assert.equal(at(consumed, 100, 150, 2002, 2007, 4000, false, false), false);
  assert.equal(at(consumed, 100, 851, 2002, 2007, 4000, false, true), false);
  assert.equal(at(consumed, 100, 99, 2002, 2007, 4000, false, true), false);
  assert.equal(at(initialAgentReaderScrollIntent(), 100, 150, 2002, 2007, 4000, false, true), false);
  assert.equal(at(gesture, 100, 150, 2002, 2007, 4000, false, true), false);
});

test("the renderer reconciles a recent physical edge with a snap but never grants passive seeks elsewhere", () => {
  const source = readSrc("components/assistant-ui/agent-history-scroll-messages.tsx");
  assert.match(source, /physicalEdgeReconcile\(\)/);
  assert.match(source, /if \(changed && freshGesture\)/);
  assert.match(source, /recentPhysicalEdge \? "bottom" : undefined/);
  assert.match(source, /lastReaderGestureAt\.current = performance\.now\(\)/);
});

test("streaming tail follows the canonical viewport owner, not a stale gesture cache", () => {
  const autoscroll = readSrc("components/assistant-ui/use-intent-aware-autoscroll.tsx");
  const renderer = readSrc("components/assistant-ui/agent-history-scroll-messages.tsx");
  assert.match(autoscroll, /getIsFollowingBottom = useCallback/);
  assert.match(renderer, /const followingOnAppend = getIsFollowingBottom\(\)/);
  assert.match(renderer, /if \(getIsFollowingBottom\(\)\) \{\s*following\.current = true;/);
  assert.match(renderer, /if \(inGutter\) interrupt\(\)/);
});


test("DOM budget rebalance follows viewport tail intent during tool-output streaming", () => {
  const renderer = readSrc("components/assistant-ui/agent-history-scroll-messages.tsx");
  assert.match(renderer, /const followingTail = getIsFollowingBottom\(\)/);
  assert.match(renderer, /followingTail \? countRef\.current - 1 : visibleIndex\.current/);
  assert.match(renderer, /if \(followingTail\) jump\.current = "bottom"/);
});
