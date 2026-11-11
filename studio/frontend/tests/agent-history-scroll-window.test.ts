// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";
import { AgentHeightIndex, agentIndexAtScrollPosition, latestAgentWindow, agentWindowAtIndex, agentWindowOnAppend, nextProgressiveAgentWindow, agentViewportCoverageRecoveryTarget, AGENT_DEFAULT_ROWS, AGENT_INITIAL_ROWS, AGENT_MAX_ROWS, AGENT_WINDOW_SIZES } from "../src/components/assistant-ui/agent-history-scroll-window.ts";

test("new agent mode defaults to five while explicit 32 preserves legacy 96-row append headroom", () => {
  assert.equal(AGENT_DEFAULT_ROWS, 5);
  assert.equal(AGENT_INITIAL_ROWS, 32);
  assert.ok(AGENT_WINDOW_SIZES.includes(AGENT_DEFAULT_ROWS));
  assert.deepEqual(latestAgentWindow(400, AGENT_DEFAULT_ROWS), { start: 395, end: 400 });
  assert.deepEqual(latestAgentWindow(400, AGENT_INITIAL_ROWS), { start: 368, end: 400 });
  assert.deepEqual(agentWindowOnAppend({ start: 368, end: 400 }, 400, 401, true, 32), { start: 368, end: 401 });
  assert.deepEqual(agentWindowOnAppend({ start: 395, end: 400 }, 400, 401, true, 5), { start: 396, end: 401 });
});

test("first and last 32 messages are reachable without paging buttons", () => {
  assert.deepEqual(latestAgentWindow(1000), { start: 968, end: 1000 });
  assert.deepEqual(agentWindowAtIndex(1000, 0), { start: 0, end: 32 });
  assert.deepEqual(agentWindowAtIndex(1000, 999), { start: 968, end: 1000 });
});

test("scrolling across every window never loses or duplicates indices", () => {
  for (let count = 1; count < 300; count++) {
    let window = latestAgentWindow(count);
    for (let index = count - 1; index >= 0; index--) {
      window = agentWindowAtIndex(count, index, window);
      assert.ok(index >= window.start && index < window.end);
      assert.ok(window.end - window.start <= AGENT_MAX_ROWS);
    }
    for (let index = 0; index < count; index++) {
      window = agentWindowAtIndex(count, index, window);
      assert.ok(index >= window.start && index < window.end);
    }
  }
});

test("append preserves the mounted window on its first update", () => {
  let window = latestAgentWindow(100);
  assert.deepEqual(agentWindowOnAppend(window, 100, 101, true), { start: 68, end: 101 });
  let count = 100;
  for (let i = 0; i < 250; i++) {
    window = agentWindowOnAppend(window, count, count + 1, true);
    count++;
    assert.equal(window.end, count);
    assert.ok(window.end - window.start <= AGENT_MAX_ROWS);
  }
  assert.deepEqual(agentWindowOnAppend({ start: 30, end: 62 }, 100, 101, false), { start: 30, end: 62 });
});

test("variable-height rows contribute their measured height to all offsets", () => {
  const sizes = [32, 4000, 21000, 0, 240, 100, 9300, 10, 2300, 5];
  const heights = new AgentHeightIndex(sizes.length, 500);
  sizes.forEach((size, index) => heights.measure(index, size));
  for (let k = 0; k <= sizes.length; k++) {
    assert.equal(heights.offset(k), sizes.slice(0, k).reduce((a, b) => a + b, 0));
  }
  assert.equal(heights.indexAt(heights.total), sizes.length - 1);
});

test("growing and shrinking the transcript preserves earlier measurement", () => {
  const heights = new AgentHeightIndex(10_000, 500);
  heights.measure(9980, 20_000);
  const original = heights.total;
  heights.resize(20_000);
  assert.equal(heights.offset(10_000), original);
  heights.measure(12_345, 1800);
  assert.equal(heights.indexAt(heights.offset(12_345) + 1), 12_345);
  heights.resize(9000);
  assert.equal(heights.total, 9000 * 500);
});

test("invalid indices are rejected rather than crashing the chat", () => {
  assert.throws(() => new AgentHeightIndex(-1), RangeError);
  const heights = new AgentHeightIndex(0);
  assert.equal(heights.indexAt(100), 0);
  heights.resize(1);
  assert.equal(heights.measure(10, 50), false);
});


test("scroll position uses the top spacer as document origin even at a distant window", () => {
  const heights = new AgentHeightIndex(1_000, 720);
  const viewHeight = 600;
  const viewTop = 80;
  for (const index of [0, 31, 100, 500, 967, 998]) {
    const scrollTop = index * 720;
    // The spacer starts at the document origin, even when the mounted rows
    // begin at index 968 and its height is 968 * 720.
    const spacerTop = viewTop - scrollTop;
    assert.equal(agentIndexAtScrollPosition(heights, spacerTop, viewTop, scrollTop, viewHeight), index);
  }
  // A measured 20k-px tool message moves every later row's offset.
  heights.measure(400, 20_000);
  const scrollTop = heights.offset(500);
  assert.equal(agentIndexAtScrollPosition(heights, viewTop - scrollTop, viewTop, scrollTop, viewHeight), 500);
});

test("2/5/8/16/32 windows reach every index forward and backward", () => {
  for (const rows of AGENT_WINDOW_SIZES) {
    for (const count of [1, 2, 5, 6, 31, 32, 33, 400, 4000]) {
      let window = latestAgentWindow(count, rows);
      assert.equal(window.end - window.start, Math.min(count, rows));
      for (let index = count - 1; index >= 0; index--) {
        window = agentWindowAtIndex(count, index, window, rows);
        assert.ok(index >= window.start && index < window.end, `${rows}: up ${index}`);
        assert.ok(window.end - window.start <= rows);
      }
      for (let index = 0; index < count; index++) {
        window = agentWindowAtIndex(count, index, window, rows);
        assert.ok(index >= window.start && index < window.end, `${rows}: down ${index}`);
        assert.ok(window.end - window.start <= rows);
      }
      for (const changedTo of AGENT_WINDOW_SIZES) {
        const changed = agentWindowAtIndex(count, Math.min(count - 1, 3), undefined, changedTo);
        assert.ok(changed.end - changed.start <= changedTo);
      }
    }
  }
});

test("small windows do not expand during append, legacy 32 may expand to 96", () => {
  for (const size of AGENT_WINDOW_SIZES) {
    let window = latestAgentWindow(100, size);
    let count = 100;
    for (let i = 0; i < 150; i++) {
      window = agentWindowOnAppend(window, count, count + 1, true, size);
      count++;
      assert.equal(window.end, count);
      assert.ok(window.end - window.start <= (size === 32 ? AGENT_MAX_ROWS : size));
    }
    assert.deepEqual(agentWindowOnAppend(window, count, count + 1, false, size), window);
  }
});

test("distant passive seeks admit at most two new rows per frame", () => {
  const destination = agentWindowAtIndex(1000, 250, undefined, 5);
  let current = { start: 250, end: 251 };
  assert.ok(destination.start <= 250 && destination.end > 250);
  let steps = 0;
  while (current.start !== destination.start || current.end !== destination.end) {
    const next = nextProgressiveAgentWindow(current, destination);
    const admitted = (next.end - next.start) - (current.end - current.start);
    assert.ok(admitted >= 1 && admitted <= 2);
    assert.ok(next.start <= 250 && next.end > 250);
    current = next;
    steps++;
    assert.ok(steps <= 2);
  }
  assert.deepEqual(current, destination);
  assert.equal(steps, 2);
  assert.deepEqual(nextProgressiveAgentWindow(current, destination), destination);
  assert.deepEqual(nextProgressiveAgentWindow({ start: 250, end: 251 }, destination, 0),
    { start: 250, end: 251 });
});

test("an interrupted progressive window cannot be accepted as a finished K=5 window", () => {
  const final = agentWindowAtIndex(1000, 250, undefined, 5);
  for (const partial of [
    { start: 250, end: 251 },
    { start: 249, end: 252 },
    { start: 248, end: 252 },
  ]) {
    const resumed = agentWindowAtIndex(1000, 250, partial, 5);
    assert.deepEqual(resumed, final);
    assert.equal(resumed.end - resumed.start, 5);
  }
  assert.deepEqual(agentWindowAtIndex(1000, 250, final, 5), final);
});


test("virtualized agent rows inherit upstream stale-safe message-index resolver", () => {
  const source = readSrc("components/assistant-ui/agent-history-scroll-messages.tsx");
  assert.match(source, /import \{ MessageByIndexProvider \} from "\.\/stale-safe-scopes";/);
  assert.doesNotMatch(source, /import \{[^\n]*MessageByIndexProvider[^\n]*\} from "@assistant-ui\/react"/);
  assert.match(source, /<MessageByIndexProvider index=\{index\}>/);
});

test("explicit fixed 32 keeps exactly 32 mounted rows across 300 appends and seeks", () => {
  let count = 400;
  let range = latestAgentWindow(count, 32);
  for (let i = 0; i < 300; i++) {
    range = agentWindowOnAppend(range, count, count + 1, true, 32, true);
    count += 1;
    assert.equal(range.end, count);
    assert.equal(range.end - range.start, 32);
  }
  for (let target = 0; target < count; target += 13) {
    range = agentWindowAtIndex(count, target, range, 32, true);
    assert.ok(range.start <= target && target < range.end);
    assert.equal(range.end - range.start, 32);
  }
  assert.deepEqual(agentWindowOnAppend(range, count, count + 1, false, 32, true), range);
});

test("physical viewport rescue selects an off-window row despite a tall mounted window", () => {
  const atTail = { start: 384, end: 400 };
  // The real 240px-wheel Chromium failure had 16 rows mounted, but the first
  // actual DOM row began 661px below the 960px physical viewport top.
  assert.equal(
    agentViewportCoverageRecoveryTarget(atTail, 383, 386, 661, 0, 960),
    383,
  );
  // A faster forward wheel left 535px uncovered below the final mounted row.
  assert.equal(
    agentViewportCoverageRecoveryTarget({ start: 375, end: 391 }, 377, 391, 0, 535, 960),
    391,
  );
  // Never retarget for a small gap, an already mounted target, or invalid data.
  assert.equal(agentViewportCoverageRecoveryTarget(atTail, 383, 399, 70, 0, 960), null);
  assert.equal(agentViewportCoverageRecoveryTarget(atTail, 384, 399, 661, 0, 960), null);
  assert.equal(agentViewportCoverageRecoveryTarget(atTail, 383, 399, 661, 0, NaN), null);
});
