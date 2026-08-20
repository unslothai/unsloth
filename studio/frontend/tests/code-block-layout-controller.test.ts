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
  CODE_BLOCK_SELECTOR,
  CODE_BLOCK_SETTLE_MS,
  type CodeBlockLayout,
  addedACodeBlock,
  createCodeBlockLayoutController,
  createCodeBlockRemountWatcher,
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

test("a remount on a quiet thread takes the hold back and measures again", () => {
  // Leaving the edit textarea on a COMPLETED reply swaps the message body back for its rendered
  // parts, so every code block in it is a new element with no last remembered size -- on a
  // thread that never stopped being quiet, and so is already released. Measured in Chromium 151:
  // released, such a block lays out at the 200px fallback for one frame on a 2,169px block.
  const { clock, controller, seen } = build();
  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  clock.flushTimeouts();
  assert.equal(controller.layout(), "settled");

  controller.remeasure();
  assert.equal(
    controller.layout(),
    "building",
    "the hold is immediate: the new elements must not be skippable on the frame they appear",
  );
  assert.deepEqual(seen, ["settled", "building"]);

  // And it releases again on its own, with the same frame pair and delay a run gets. A remount
  // that only held would leave the thread paying for every block for the rest of its life.
  assert.equal(clock.pendingTimeouts(), 0, "the frames come first here too");
  clock.flushFrame();
  clock.flushFrame();
  assert.deepEqual(clock.flushTimeouts(), [900]);
  assert.equal(controller.layout(), "settled");
  assert.deepEqual(seen, ["settled", "building", "settled"]);
});

test("a remount restarts the settle window rather than riding out the one already armed", () => {
  // Not idempotent, unlike setRunning: two edits saved in quick succession are two remounts, and
  // the second one's elements need the full window from when THEY were created.
  const { clock, controller } = build();
  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  assert.equal(clock.pendingTimeouts(), 1);

  controller.remeasure();
  assert.equal(
    clock.pendingTimeouts(),
    0,
    "the release armed before the remount must be dropped",
  );
  assert.equal(clock.pendingFrames(), 1, "and a fresh frame pair started");
  clock.flushFrame();
  clock.flushFrame();
  clock.flushTimeouts();
  assert.equal(controller.layout(), "settled");
});

test("a remount during a run holds without arming a release of its own", () => {
  // The store that drives this is not the run state, so a remount can land mid-stream. Arming a
  // release from here would settle the thread in the middle of the reply that is still writing
  // into it; the release stays with the run, which arms one when it ends.
  const { clock, controller, seen } = build();
  controller.setRunning(true);
  controller.remeasure();
  assert.equal(controller.layout(), "building");
  assert.equal(clock.pendingFrames(), 0, "no frame pair while the run is live");
  assert.equal(clock.pendingTimeouts(), 0);

  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  clock.flushTimeouts();
  assert.equal(controller.layout(), "settled");
  assert.deepEqual(seen, ["settled"]);
});

test("disposing cancels a release armed by a remount", () => {
  const { clock, controller } = build();
  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  clock.flushTimeouts();
  controller.remeasure();
  clock.flushFrame();
  clock.flushFrame();
  assert.equal(clock.pendingTimeouts(), 1);
  controller.dispose();
  assert.equal(clock.pendingTimeouts(), 0);
  clock.flushTimeouts();
  assert.equal(controller.layout(), "building");
});

// ── the remount watcher ──────────────────────────────────────────────────────────────────────
//
// The controller only knows about runs. What tells it that blocks were re-created on a thread
// that never stopped being quiet is a MutationObserver on the thread root, and the policy for
// when that observer is connected is the part worth pinning down: connected only while settled,
// disconnected before it reports anything, re-armed only when the controller settles again.
// Every test below is a way of getting that wrong that costs either a flicker or a live loop.

/** A node stand-in. The real predicate is "is, or contains, a code block"; here it is a flag. */
type FakeNode = { block: boolean };
const isBlock = (node: FakeNode): boolean => node.block;
const records = (...batches: FakeNode[][]) =>
  batches.map((addedNodes) => ({ addedNodes }));

function buildWatcher() {
  const calls: string[] = [];
  const watcher = createCodeBlockRemountWatcher({
    connect: () => calls.push("connect"),
    disconnect: () => calls.push("disconnect"),
    onRemount: () => calls.push("remount"),
  });
  return { calls, watcher };
}

test("a batch is only interesting if something in it is, or contains, a code block", () => {
  // The containment half is why the predicate is injected rather than a marker check: React
  // mounts a reply's rendered parts as ONE added node, with the blocks beneath it. A scan that
  // asked only whether the added node itself was a block would answer no on every real path.
  assert.equal(addedACodeBlock(records([{ block: false }]), isBlock), false);
  assert.equal(addedACodeBlock(records([]), isBlock), false);
  assert.equal(addedACodeBlock(records([], []), isBlock), false);
  assert.equal(
    addedACodeBlock(records([{ block: false }], [{ block: true }]), isBlock),
    true,
    "the block can be anywhere in the batch, not just the first record",
  );
  // The selector is shared with index.css and with streamdown's own markup, so a rename that
  // only touched one of the two would silently stop matching anything.
  assert.equal(CODE_BLOCK_SELECTOR, '[data-streamdown="code-block"]');
});

test("the watcher observes only while the thread is settled", () => {
  // During a run the hold is already on, so there is nothing to take back -- and a childList
  // observer with subtree:true on a thread root fires on every streaming mutation there is.
  const { calls, watcher } = buildWatcher();
  assert.equal(
    watcher.connected(),
    false,
    "a controller starts held, so this starts off",
  );

  watcher.layoutChanged("building");
  assert.deepEqual(
    calls,
    [],
    "already off; connecting is not toggled for its own sake",
  );

  watcher.layoutChanged("settled");
  assert.deepEqual(calls, ["connect"]);
  watcher.layoutChanged("settled");
  assert.deepEqual(calls, ["connect"], "a repeated state does not re-observe");

  watcher.layoutChanged("building");
  assert.deepEqual(calls, ["connect", "disconnect"]);
});

test("a mutation that creates no code block leaves the hold alone", () => {
  // A settled thread still mutates: an action bar mounting under the cursor, a tooltip, a
  // branch counter. Taking the hold back for those would put every reply in the thread back
  // under `content-visibility: visible` each time the mouse crossed one.
  const { calls, watcher } = buildWatcher();
  watcher.layoutChanged("settled");
  watcher.sawMutations(records([{ block: false }, { block: false }]), isBlock);
  assert.deepEqual(calls, ["connect"]);
  assert.equal(
    watcher.connected(),
    true,
    "and it stays armed for the one that matters",
  );
});

test("the first qualifying batch disconnects BEFORE it reports the remount", () => {
  // The disconnect is what makes "can this re-trigger itself" answerable without reasoning
  // about what onRemount does to the DOM. Order matters: reporting first would leave a window
  // in which the observer is live during the handler.
  const { calls, watcher } = buildWatcher();
  watcher.layoutChanged("settled");
  watcher.sawMutations(records([{ block: true }]), isBlock);
  assert.deepEqual(calls, ["connect", "disconnect", "remount"]);
  assert.equal(watcher.connected(), false);
});

test("a batch delivered after the disconnect does not spend a second remeasure", () => {
  // An observer can still deliver records for mutations that happened before disconnect() was
  // called, and the second reply of a two-reply commit is exactly that shape.
  const { calls, watcher } = buildWatcher();
  watcher.layoutChanged("settled");
  watcher.sawMutations(records([{ block: true }]), isBlock);
  watcher.sawMutations(records([{ block: true }]), isBlock);
  assert.deepEqual(calls, ["connect", "disconnect", "remount"]);
});

test("the watcher terminates: driving it from a real controller reaches a fixed point", () => {
  // The loop this is guarding against is remeasure() -> "building" -> DOM change -> observer ->
  // remeasure(). Wire the two together for real, feed it one remount, and run the clock out: it
  // has to come to rest settled and armed, having spent exactly one remeasure.
  const clock = fakeClock();
  const seen: CodeBlockLayout[] = [];
  let observing = false;
  let remeasures = 0;
  const controller = createCodeBlockLayoutController({
    settleMs: 900,
    timers: clock.timers,
    onChange: (layout) => {
      seen.push(layout);
      watcher.layoutChanged(layout);
    },
  });
  const watcher = createCodeBlockRemountWatcher({
    connect: () => {
      observing = true;
    },
    disconnect: () => {
      observing = false;
    },
    onRemount: () => {
      remeasures += 1;
      controller.remeasure();
      // What remeasure() actually does to the DOM is one attribute write on the observed root.
      // The observer is registered for childList only, so feeding an attribute-shaped batch
      // back in stands for that write: it must not be seen, and it must not requeue anything.
      watcher.sawMutations(records([{ block: false }]), isBlock);
    },
  });

  controller.setRunning(false);
  clock.flushFrame();
  clock.flushFrame();
  clock.flushTimeouts();
  assert.equal(controller.layout(), "settled");
  assert.equal(observing, true, "a settled thread is watched");

  watcher.sawMutations(records([{ block: true }]), isBlock);
  assert.equal(remeasures, 1);
  assert.equal(
    controller.layout(),
    "building",
    "the hold is back on the same tick",
  );
  assert.equal(observing, false, "and the observer is off while it is");

  clock.flushFrame();
  clock.flushFrame();
  assert.deepEqual(clock.flushTimeouts(), [900]);
  assert.equal(controller.layout(), "settled");
  assert.equal(remeasures, 1, "exactly one remeasure for one remount");
  assert.equal(observing, true, "re-armed by settling, and by nothing else");
  assert.equal(
    clock.pendingFrames(),
    0,
    "nothing left queued: this is a fixed point",
  );
  assert.equal(clock.pendingTimeouts(), 0);
  assert.deepEqual(seen, ["settled", "building", "settled"]);
});

test("a remount seen mid-run neither re-arms the watcher nor settles the thread", () => {
  // Streaming commits add code blocks constantly. If the watcher were live during a run every
  // one of them would land here, and an armed release from any of them would settle the thread
  // inside the reply still writing into it.
  const clock = fakeClock();
  let observing = false;
  const controller = createCodeBlockLayoutController({
    settleMs: 900,
    timers: clock.timers,
    onChange: (layout) => {
      watcher.layoutChanged(layout);
    },
  });
  const watcher = createCodeBlockRemountWatcher({
    connect: () => {
      observing = true;
    },
    disconnect: () => {
      observing = false;
    },
    onRemount: () => {
      controller.remeasure();
    },
  });

  controller.setRunning(true);
  assert.equal(observing, false);
  // Even if a stale batch arrives, it is refused at the gate rather than by the controller.
  watcher.sawMutations(records([{ block: true }]), isBlock);
  assert.equal(clock.pendingFrames(), 0);
  assert.equal(clock.pendingTimeouts(), 0);
  assert.equal(controller.layout(), "building");
});

test("disposing the watcher disconnects it", () => {
  const { calls, watcher } = buildWatcher();
  watcher.layoutChanged("settled");
  watcher.dispose();
  assert.deepEqual(calls, ["connect", "disconnect"]);
  watcher.dispose();
  assert.deepEqual(
    calls,
    ["connect", "disconnect"],
    "and disposing twice is quiet",
  );
});

test("the shipped settle delay outlasts a frame at 60Hz by a wide margin", () => {
  // The delay has to cover the render that finalizes a message, which lands one or two frames
  // after the stream ends. Anything at or below a frame would release inside it.
  assert.ok(
    CODE_BLOCK_SETTLE_MS >= 500,
    `settle delay ${CODE_BLOCK_SETTLE_MS}ms is too short to outlast a finalizing render`,
  );
});
