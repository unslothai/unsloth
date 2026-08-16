// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// When a thread is allowed to let its code blocks skip their own rendering.
//
// The rule the controller enforces is that a code block must have been laid out at its real
// height at least once before `content-visibility: auto` is allowed to skip it, because an
// element that has never been rendered has no last remembered size and is skipped at the 200px
// `contain-intrinsic-size` fallback instead. Every test below is written against a way of
// getting that wrong that was reachable while this was built.

import assert from "node:assert/strict";
import test from "node:test";

import {
  CODE_BLOCK_LAYOUT_ATTRIBUTE,
  CODE_BLOCK_SETTLE_MS,
  type CodeBlockLayout,
  createCodeBlockLayoutController,
} from "../src/components/assistant-ui/code-block-layout.ts";

/** A hand-cranked clock, so the tests assert on the ORDER of events rather than on wall time. */
function fakeClock() {
  let nextHandle = 1;
  const timeouts = new Map<number, { callback: () => void; ms: number }>();
  const frames = new Map<number, () => void>();
  return {
    timers: {
      setTimeout: (callback: () => void, ms: number) => {
        const handle = nextHandle++;
        timeouts.set(handle, { callback, ms });
        return handle;
      },
      clearTimeout: (handle: number) => {
        timeouts.delete(handle);
      },
      requestAnimationFrame: (callback: () => void) => {
        const handle = nextHandle++;
        frames.set(handle, callback);
        return handle;
      },
      cancelAnimationFrame: (handle: number) => {
        frames.delete(handle);
      },
    },
    pendingTimeouts: () => timeouts.size,
    pendingFrames: () => frames.size,
    /** Run every frame callback currently queued, once. */
    flushFrame: () => {
      const queued = [...frames.entries()];
      frames.clear();
      for (const [, callback] of queued) callback();
    },
    /** Run every timeout currently queued, and report the delays they were armed with. */
    flushTimeouts: (): number[] => {
      const queued = [...timeouts.entries()];
      timeouts.clear();
      for (const [, entry] of queued) entry.callback();
      return queued.map(([, entry]) => entry.ms);
    },
  };
}

function build(settleMs = 900) {
  const clock = fakeClock();
  const seen: CodeBlockLayout[] = [];
  const controller = createCodeBlockLayoutController({
    settleMs,
    timers: clock.timers,
    onChange: (layout) => seen.push(layout),
  });
  return { clock, seen, controller };
}

test("a thread starts held, before anything has told it whether it is running", () => {
  // A controller that started settled would let a mounting thread skip blocks that no frame has
  // measured yet, which is the exact case the hold exists for. The attribute name is exported
  // with it because index.css keys off that spelling.
  const { controller, seen } = build();
  assert.equal(controller.layout(), "building");
  assert.deepEqual(seen, []);
  assert.equal(CODE_BLOCK_LAYOUT_ATTRIBUTE, "data-code-block-layout");
});

test("a quiet thread releases, but only after a frame pair and then the settle delay", () => {
  const { clock, controller, seen } = build(900);
  controller.setRunning(false);

  // Two frames first. A release armed from inside the rendering update that created a block
  // could otherwise expire before that block had ever been through layout.
  assert.equal(controller.layout(), "building");
  assert.equal(
    clock.pendingTimeouts(),
    0,
    "the clock must not start before the frames",
  );
  clock.flushFrame();
  assert.equal(clock.pendingTimeouts(), 0, "one frame is not two");
  clock.flushFrame();
  assert.equal(clock.pendingTimeouts(), 1);

  assert.equal(
    controller.layout(),
    "building",
    "still held while the delay runs",
  );
  const delays = clock.flushTimeouts();
  assert.deepEqual(delays, [900], "armed with the settle delay it was given");
  assert.equal(controller.layout(), "settled");
  assert.deepEqual(
    seen,
    ["settled"],
    "one notification, on the transition only",
  );
});

test("a running thread is held, and holds again the moment a new run starts", () => {
  const { clock, controller, seen } = build();
  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  clock.flushTimeouts();
  assert.equal(controller.layout(), "settled");

  controller.setRunning(true);
  assert.equal(
    controller.layout(),
    "building",
    "the hold is immediate: a block created by this run must not be skippable first",
  );
  assert.deepEqual(seen, ["settled", "building"]);
});

test("a run that starts inside the settle window cancels the release", () => {
  // The window is a delay, not a deadline. A reply that arrives while it is counting down must
  // not be released mid-stream just because the clock was already running.
  const { clock, controller, seen } = build();
  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  assert.equal(clock.pendingTimeouts(), 1);

  controller.setRunning(true);
  assert.equal(
    clock.pendingTimeouts(),
    0,
    "the pending release must be cancelled, not left",
  );
  clock.flushTimeouts();
  assert.equal(controller.layout(), "building");
  assert.deepEqual(
    seen,
    [],
    "it never left the held state, so nothing changed",
  );
});

test("a run that starts before the frame pair completes cancels the release too", () => {
  const { clock, controller } = build();
  controller.setRunning(false);
  clock.flushFrame();
  assert.equal(clock.pendingFrames(), 1);

  controller.setRunning(true);
  assert.equal(clock.pendingFrames(), 0, "the queued frame must be cancelled");
  clock.flushFrame();
  clock.flushTimeouts();
  assert.equal(controller.layout(), "building");
});

test("repeating the running state does not restart the settle delay", () => {
  // The signal feeds this from a store subscription, which republishes the same value freely. A
  // controller that re-armed on every repeat would never reach the end of its own delay on a
  // thread whose store is busy.
  const { clock, controller } = build();
  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  assert.equal(clock.pendingTimeouts(), 1);

  controller.setRunning(false);
  controller.setRunning(false);
  assert.equal(
    clock.pendingTimeouts(),
    1,
    "still the same single armed release",
  );
  assert.equal(clock.pendingFrames(), 0, "and no second frame pair was queued");
  clock.flushTimeouts();
  assert.equal(controller.layout(), "settled");
});

test("a second quiet run after a release does not notify again", () => {
  const { clock, controller, seen } = build();
  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  clock.flushTimeouts();
  controller.setRunning(true);
  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  clock.flushTimeouts();
  assert.deepEqual(seen, ["settled", "building", "settled"]);
});

test("disposing cancels a release that has not fired", () => {
  // The signal disposes on unmount. A release that fired afterwards would write the attribute
  // onto a detached root, and on a remount the thread would come back released.
  const { clock, controller, seen } = build();
  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  controller.dispose();
  assert.equal(clock.pendingTimeouts(), 0);
  clock.flushTimeouts();
  assert.equal(controller.layout(), "building");
  assert.deepEqual(seen, []);
});

test("the shipped settle delay outlasts a frame at 60Hz by a wide margin", () => {
  // The delay has to cover the render that finalizes a message, which lands one or two frames
  // after the stream ends. Anything at or below a frame would release inside it.
  assert.ok(
    CODE_BLOCK_SETTLE_MS >= 500,
    `settle delay ${CODE_BLOCK_SETTLE_MS}ms is too short to outlast a finalizing render`,
  );
});
