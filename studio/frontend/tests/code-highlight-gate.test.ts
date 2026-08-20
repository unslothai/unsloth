// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// When a code block is allowed to give up its Shiki spans.
//
// The rule the gate enforces is that a block renders plain only when something has MEASURED it
// outside the viewport plus the buffer, and it is not being streamed into. Both escape hatches
// point the same way, towards colour, because the failure they prevent -- a visibly uncoloured
// block, or worse a block decoloured under a scrolling user -- is far worse than the slow standing
// DOM this exists to shrink. Every test below is written against a way of getting that backwards
// that was reachable while this was built.

import assert from "node:assert/strict";
import test from "node:test";

import {
  CODE_HIGHLIGHT_BUFFER_PX,
  CODE_HIGHLIGHT_STREAMING_GRACE_MS,
  createCodeHighlightGate,
  decideHighlightMode,
} from "../src/components/assistant-ui/code-highlight-gate.ts";

const VIEWPORT = 900;

/** A gate on a hand-cranked clock, so the tests assert on ORDER rather than on wall time. */
function build(options: { bufferPx?: number; graceMs?: number } = {}) {
  let clock = 10_000;
  const seen: string[] = [];
  const gate = createCodeHighlightGate({
    bufferPx: options.bufferPx ?? 100,
    streamingGraceMs: options.graceMs ?? CODE_HIGHLIGHT_STREAMING_GRACE_MS,
    viewportHeight: VIEWPORT,
    now: () => clock,
  });
  gate.subscribe((id) => seen.push(id));
  return {
    gate,
    seen,
    tick: (ms: number) => {
      clock += ms;
    },
    at: () => clock,
  };
}

test("a block nobody has located yet is highlighted, not plain", () => {
  // This is the case the whole mechanism passes through on first render: the element does not
  // exist until a result has been rendered, so there is no geometry to have. A gate that defaulted
  // to plain would decolour every block in the thread for the frame before it was measured,
  // including the ones the user is looking at.
  assert.equal(
    decideHighlightMode({
      geometry: null,
      viewportHeight: VIEWPORT,
      bufferPx: CODE_HIGHLIGHT_BUFFER_PX,
      streaming: false,
    }),
    "highlighted",
  );
  const { gate } = build();
  assert.equal(gate.mode("never-seen"), "highlighted");
});

test("the band is the viewport plus the buffer on BOTH sides, and its edges are inclusive", () => {
  const decide = (top: number, bottom: number) =>
    decideHighlightMode({
      geometry: { top, bottom },
      viewportHeight: VIEWPORT,
      bufferPx: 100,
      streaming: false,
    });

  assert.equal(decide(10, 200), "highlighted", "plainly on screen");
  // Above: a block ends exactly at the top edge of the buffer.
  assert.equal(
    decide(-300, -100),
    "highlighted",
    "touching the top edge is inside",
  );
  assert.equal(decide(-300, -101), "plain", "one pixel past it is outside");
  // Below: a block starts exactly at the bottom edge of the buffer.
  assert.equal(
    decide(VIEWPORT + 100, VIEWPORT + 400),
    "highlighted",
    "touching the bottom edge",
  );
  assert.equal(
    decide(VIEWPORT + 101, VIEWPORT + 400),
    "plain",
    "one pixel past it",
  );
  // A block taller than the viewport, scrolled so neither end is in it, is still on screen.
  assert.equal(
    decide(-5000, 5000),
    "highlighted",
    "a block spanning the viewport is in view",
  );
});

test("a streaming block stays highlighted wherever it is measured to be", () => {
  // The block being streamed is the reply the user is waiting on, and it owns the plugin's
  // incremental caches. A gate that could take its colours off mid-stream would be fighting the
  // one path in the plugin that is already fast.
  assert.equal(
    decideHighlightMode({
      geometry: { top: 99_999, bottom: 100_999 },
      viewportHeight: VIEWPORT,
      bufferPx: CODE_HIGHLIGHT_BUFFER_PX,
      streaming: true,
    }),
    "highlighted",
  );
});

test("the streaming hold expires on the clock, with no event to expire it", () => {
  // Nothing calls back into the gate when a stream ENDS -- the last chunk is just the last call.
  // A hold that needed an event to clear it would pin every block that ever streamed.
  const { gate, tick } = build({ graceMs: 1000 });
  gate.place("a", { top: 99_999, bottom: 100_999 });
  assert.equal(gate.mode("a"), "plain", "far away and not streaming");

  gate.markStreaming("a");
  assert.equal(gate.mode("a"), "highlighted", "held by the stream");
  tick(999);
  assert.equal(gate.mode("a"), "highlighted", "still inside the grace window");
  tick(2);
  assert.equal(
    gate.mode("a"),
    "plain",
    "the window passed, with no event to say so",
  );
});

test("a block registered by place() is not treated as having streamed at time zero", () => {
  // `lastGrowthAt` starting at 0 rather than -Infinity reads as "grew at the epoch", which is
  // inside the grace window for the first second of a page's life and nowhere near it afterwards.
  // That is a gate whose behaviour depends on how long the app has been open.
  const gate = createCodeHighlightGate({
    bufferPx: 100,
    viewportHeight: VIEWPORT,
    now: () => 5, // 5ms after start, well inside any grace window
  });
  gate.place("a", { top: 99_999, bottom: 100_999 });
  assert.equal(gate.mode("a"), "plain");
});

test("listeners fire on the transition only, never on a repeat", () => {
  // The plugin turns each notification into a re-render of that block. A gate that republished the
  // current mode on every scroll event would re-render every mounted code block on every frame,
  // which is the cost this exists to remove.
  const { gate, seen } = build();
  gate.place("a", { top: 10, bottom: 200 });
  assert.deepEqual(seen, [], "already highlighted, so nothing changed");

  gate.place("a", { top: 5000, bottom: 5200 });
  assert.deepEqual(seen, ["a"], "highlighted -> plain");

  gate.place("a", { top: 5100, bottom: 5300 });
  assert.deepEqual(seen, ["a"], "still plain, still one notification");

  gate.place("a", { top: 20, bottom: 220 });
  assert.deepEqual(seen, ["a", "a"], "plain -> highlighted");
});

test("forgetting a position is not the same as reporting it far away", () => {
  // The plugin calls `forget` when it drops a fence. That must not be a downgrade: it means the
  // gate no longer knows anything about the block, which reads as highlighted.
  const { gate } = build();
  gate.place("a", { top: 5000, bottom: 5200 });
  assert.equal(gate.mode("a"), "plain");
  gate.forget("a");
  assert.equal(gate.mode("a"), "highlighted");
});

test("a shorter viewport re-settles every block it already knows about", () => {
  // The band is measured from the viewport height, so a window resize moves it for blocks that
  // have not moved and will not fire an intersection of their own.
  const { gate, seen } = build();
  gate.place("a", { top: 880, bottom: 990 });
  assert.equal(gate.mode("a"), "highlighted", "inside 900 + 100");
  gate.setViewportHeight(400);
  assert.equal(gate.mode("a"), "plain", "outside 400 + 100");
  assert.deepEqual(seen, ["a"]);
});

test("setting the same viewport height again re-settles nothing", () => {
  const { gate, seen } = build();
  gate.place("a", { top: 10, bottom: 200 });
  gate.setViewportHeight(VIEWPORT);
  assert.deepEqual(seen, []);
});

test("announcing is how a block gets found, and it does not decide anything", () => {
  // The plugin announces from inside its return path, before React has committed the result. The
  // block is therefore still unmeasured at that moment, and must read as highlighted.
  const { gate } = build();
  const announced: string[] = [];
  gate.onAnnounce((id) => announced.push(id));
  gate.announce("a");
  assert.deepEqual(announced, ["a"]);
  assert.equal(gate.mode("a"), "highlighted");
});

test("the shipped buffer covers a hard flick at the measured re-highlight latency", () => {
  // Re-highlighting one fence of the 100K-rung corpus from a cold fence cache, grammars already
  // loaded, measured over all 57 content fences: median 12.7ms, p90 28.5ms, max 39.4ms, at a
  // median 1,783 characters. Such a fence renders at roughly 570px, so the buffer holds about
  // `bufferPx / 570` of them and refilling them costs that many times the p90.
  const fencePx = 570;
  const p90Ms = 28.5;
  const flickPxPerS = 5000;
  const refillMs = (CODE_HIGHLIGHT_BUFFER_PX / fencePx) * p90Ms;
  const crossingMs = (CODE_HIGHLIGHT_BUFFER_PX / flickPxPerS) * 1000;
  assert.ok(
    crossingMs > refillMs * 2,
    `buffer ${CODE_HIGHLIGHT_BUFFER_PX}px is crossed in ${crossingMs.toFixed(0)}ms at ` +
      `${flickPxPerS}px/s but needs ${refillMs.toFixed(0)}ms to refill`,
  );
});
