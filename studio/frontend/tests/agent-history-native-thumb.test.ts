// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { AgentHeightIndex } from "../src/components/assistant-ui/agent-history-scroll-window.ts";
import {
  agentNativeThumbBottomSpacerPx,
  beginAgentNativeThumbDrag,
  agentNativeThumbLiveTarget,
  resolveAgentNativeThumbRelease,
  finishAgentNativeThumbDrag,
} from "../src/components/assistant-ui/agent-history-native-thumb.ts";

const count = 400;
const viewportHeight = 960;
const heights = new AgentHeightIndex(count);
const maxScroll = heights.total - viewportHeight;
const thumbHeight = Math.max(18, viewportHeight * viewportHeight / heights.total);

function startAtBottom(pointerY = viewportHeight - thumbHeight / 2) {
  const gesture = beginAgentNativeThumbDrag({
    pointerId: 1, pointerY, viewportTop: 0, viewportHeight,
    scrollTop: maxScroll, scrollHeight: heights.total,
  });
  assert.ok(gesture, "actual scrollbar thumb must be recognized");
  return gesture;
}

test("one native thumb drag from bottom to track origin resolves to the FIRST message", () => {
  const gesture = startAtBottom();
  // Replacing 16 estimated rows by short measured rows during the drag must
  // not redefine the gesture's logical endpoint.
  const changingHeights = new AgentHeightIndex(count);
  for (let i = 380; i < 400; i++) changingHeights.measure(i, 92);
  const result = finishAgentNativeThumbDrag(gesture, {
    pointerId: 1, pointerY: 1, scrollTop: maxScroll * .3,
    viewportHeight, heights: changingHeights,
  });
  assert.deepEqual(result, { kind: "top", index: 0, topPx: 0 });
});

test("a full native thumb drag to bottom resolves to the LAST message", () => {
  const gesture = beginAgentNativeThumbDrag({
    pointerId: 4, pointerY: thumbHeight / 2,
    viewportTop: 0, viewportHeight, scrollTop: 0, scrollHeight: heights.total,
  });
  assert.ok(gesture);
  const result = finishAgentNativeThumbDrag(gesture, {
    pointerId: 4, pointerY: viewportHeight + 100, scrollTop: maxScroll * .5,
    viewportHeight, heights,
  });
  assert.deepEqual(result, { kind: "bottom", index: count - 1, topPx: maxScroll });
});

test("interior native thumb drag maps to a logical history position, not measured physical scrollTop", () => {
  const gesture = startAtBottom();
  const changed = new AgentHeightIndex(count);
  changed.measure(33, 19000);
  const targetY = viewportHeight / 2 + gesture.grabOffsetPx;
  const result = finishAgentNativeThumbDrag(gesture, {
    pointerId: 1, pointerY: targetY, scrollTop: 0,
    viewportHeight, heights: changed,
  });
  assert.equal(result?.kind, "position");
  assert.ok(result && result.index > 100 && result.index < 300, String(result?.index));
  assert.ok(result && result.topPx > changed.total * .4 && result.topPx < changed.total * .6);
});

test("scrollbar track clicks, inert thumb clicks and unrelated pointer releases are not navigation", () => {
  const bottom = startAtBottom();
  assert.equal(beginAgentNativeThumbDrag({
    pointerId: 1, pointerY: 80, viewportTop: 0, viewportHeight,
    scrollTop: maxScroll, scrollHeight: heights.total,
  }), null);
  assert.equal(finishAgentNativeThumbDrag(bottom, {
    pointerId: 2, pointerY: 0, scrollTop: 0, viewportHeight, heights,
  }), null);
  assert.equal(finishAgentNativeThumbDrag(bottom, {
    pointerId: 1, pointerY: bottom.startY,
    scrollTop: bottom.startScrollTop, viewportHeight, heights,
  }), null);
});

test("zero-range documents cannot open a native-thumb navigation transaction", () => {
  assert.equal(beginAgentNativeThumbDrag({
    pointerId: 1, pointerY: 1, viewportTop: 0, viewportHeight,
    scrollTop: 0, scrollHeight: viewportHeight,
  }), null);
});


test("drag-scoped virtual-height projection keeps native track extent fixed across row measurements", () => {
  const ledger = new AgentHeightIndex(400);
  const virtualHeight = ledger.total;
  for (const [start, end, rowHeight] of [
    [384, 400, 100], [250, 266, 110], [60, 76, 2100], [4, 20, 96],
  ]) {
    for (let i = start; i < end; i++) ledger.measure(i, rowHeight);
    // Prefix + mounted measured rows + projected unmounted suffix should
    // represent the same logical scrollbar range throughout the drag.
    const top = ledger.offset(start);
    const mounted = ledger.offset(end) - top;
    const suffix = agentNativeThumbBottomSpacerPx(virtualHeight, ledger.offset(end));
    assert.equal(top + mounted + suffix, virtualHeight);
  }
  assert.ok(ledger.total !== virtualHeight, "measurement workload must change actual height");
});

test("drag-scoped suffix is bounded if a huge row consumes all projected space", () => {
  assert.equal(agentNativeThumbBottomSpacerPx(100, 125), 0);
  assert.equal(agentNativeThumbBottomSpacerPx(100, 20), 80);
  assert.equal(agentNativeThumbBottomSpacerPx(Number.NaN, 20), 0);
});

test("R6 prioritizes the actually painted native scrollbar position over divergent pointer", () => {
  const drag = startAtBottom();
  const pointerY = viewportHeight * .52;
  const fromPointer = finishAgentNativeThumbDrag(drag, {
    pointerId: 1, pointerY, scrollTop: maxScroll * .20,
    viewportHeight, heights,
  });
  assert.ok(fromPointer && fromPointer.kind === "position");
  const actualIndex = heights.indexAt(maxScroll * .20 + viewportHeight / 3);
  const decision = resolveAgentNativeThumbRelease(fromPointer, {
    nativeScrollTop: maxScroll * .20,
    nativeScrollHeight: heights.total,
    viewportHeight, nativeIndex: actualIndex,
    rowVisibleAtNativePosition: true, count,
  });
  assert.deepEqual(decision, { kind: "keep", index: actualIndex });
  assert.ok(Math.abs(fromPointer.index - actualIndex) > 50,
    "the release must not snap from the visible native position to the cursor");
});

test("R6 preserves the visible native DOM even when heights were remeasured", () => {
  const fromPointer = { kind: "position", index: 220, topPx: maxScroll * .55 } as const;
  const actualIndex = 73;
  const decision = resolveAgentNativeThumbRelease(fromPointer, {
    nativeScrollTop: maxScroll * .18,
    nativeScrollHeight: heights.total - 12000,
    viewportHeight, nativeIndex: actualIndex,
    rowVisibleAtNativePosition: true, count,
  });
  assert.deepEqual(decision, { kind: "keep", index: actualIndex });
});

test("R6 recovers blank native viewport at its physical row, not at mouse position", () => {
  const decision = resolveAgentNativeThumbRelease({
    kind: "position", index: 300, topPx: maxScroll * .75,
  }, {
    nativeScrollTop: maxScroll * .25, nativeScrollHeight: heights.total,
    viewportHeight, nativeIndex: 101,
    rowVisibleAtNativePosition: false, count,
  });
  assert.deepEqual(decision, { kind: "recover", index: 101 });
});

test("R6 retains explicit endpoint intent and native physical endpoints", () => {
  const state = {
    nativeScrollTop: maxScroll * .4, nativeScrollHeight: heights.total,
    viewportHeight, nativeIndex: 162,
    rowVisibleAtNativePosition: true, count,
  };
  assert.deepEqual(resolveAgentNativeThumbRelease({ kind: "top", index: 0, topPx: 0 }, state),
    { kind: "top", index: 0 });
  assert.deepEqual(resolveAgentNativeThumbRelease({
    kind: "bottom", index: count - 1, topPx: maxScroll,
  }, state), { kind: "bottom", index: count - 1 });
  assert.deepEqual(resolveAgentNativeThumbRelease({
    kind: "position", index: 23, topPx: 5000,
  }, { ...state, nativeScrollTop: 0, nativeIndex: 0 }), { kind: "top", index: 0 });
  assert.deepEqual(resolveAgentNativeThumbRelease({
    kind: "position", index: 42, topPx: 15000,
  }, { ...state, nativeScrollTop: maxScroll, nativeIndex: count - 1 }),
  { kind: "bottom", index: count - 1 });
});

test("active thumb follows multiple distant ranges within the SAME held gesture", () => {
  const current = { start: 380, end: 400 };
  assert.equal(agentNativeThumbLiveTarget(current, null, 190, 400), 190);
  const staging = { start: 180, end: 196 };
  assert.equal(agentNativeThumbLiveTarget({ start: 190, end: 191 }, staging, 192, 400), null);
  assert.equal(agentNativeThumbLiveTarget({ start: 190, end: 191 }, staging, 40, 400), 40);
  assert.equal(agentNativeThumbLiveTarget({ start: 40, end: 41 }, { start: 32, end: 48 }, 44, 400), null);
  assert.equal(agentNativeThumbLiveTarget({ start: 32, end: 48 }, null, 390, 400), 390);
});

test("active native thumb never seeks for already covered rows or invalid targets", () => {
  const mounted = { start: 140, end: 156 };
  assert.equal(agentNativeThumbLiveTarget(mounted, null, 148, 400), null);
  assert.equal(agentNativeThumbLiveTarget(mounted, null, -1, 400), null);
  assert.equal(agentNativeThumbLiveTarget(mounted, null, 400, 400), null);
  assert.equal(agentNativeThumbLiveTarget(mounted, null, Number.NaN, 400), null);
  assert.equal(agentNativeThumbLiveTarget(mounted, null, 5, 0), null);
});
