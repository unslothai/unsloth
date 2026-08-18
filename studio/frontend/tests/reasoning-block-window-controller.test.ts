// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The block window driven the way a browser drives it: a commit, a measurement, a decision, a
// second commit, a layout effect. The controller needs `getBoundingClientRect`, three observers
// and four numbers off the scroll container and nothing else, so all of that is supplied here and
// the real object under test is the real object.
//
// What this file is for is the parts that only exist once geometry is involved: that the spacer is
// the exact height of what it replaced, that a block already on screen is not remounted when the
// window moves, that the reader's scroll position does not move, and that the two things which can
// invalidate a frozen height do invalidate all of them.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const BLOCK_HEIGHT = 100;
const BLOCK_COUNT = 40;
const DOC_HEIGHT = BLOCK_COUNT * BLOCK_HEIGHT;
const PANE_TOP = 50;
const PANE_HEIGHT = 256;
const PANE_WIDTH = 700;
/** Streamdown's splitter emits blocks that render nothing. This one is ours. */
const EMPTY_BLOCK = 7;

type Observed = { callback: () => void; targets: Set<unknown> };

const mutationObservers: Observed[] = [];
const resizeObservers: Observed[] = [];
const intersectionObservers: Observed[] = [];

function installObserver(sink: Observed[]) {
  return class {
    private readonly entry: Observed;
    constructor(callback: () => void) {
      this.entry = { callback, targets: new Set() };
      sink.push(this.entry);
    }
    observe(target: unknown): void {
      this.entry.targets.add(target);
    }
    unobserve(target: unknown): void {
      this.entry.targets.delete(target);
    }
    disconnect(): void {
      this.entry.targets.clear();
      const at = sink.indexOf(this.entry);
      if (at >= 0) {
        sink.splice(at, 1);
      }
    }
    takeRecords(): unknown[] {
      return [];
    }
  };
}

Object.assign(globalThis, {
  MutationObserver: installObserver(mutationObservers),
  ResizeObserver: installObserver(resizeObservers),
  IntersectionObserver: installObserver(intersectionObservers),
});

class FakeElement {
  top = 0;
  readonly children: FakeElement[] = [];
  get firstElementChild(): FakeElement | null {
    return this.children[0] ?? null;
  }
  getBoundingClientRect() {
    return {
      top: this.top,
      bottom: this.top,
      left: 0,
      right: 0,
      width: 0,
      height: 0,
      x: 0,
      y: this.top,
      toJSON: () => ({}),
    };
  }
}

class FakePane extends FakeElement {
  clientWidth = PANE_WIDTH;
  clientHeight = PANE_HEIGHT;
  scrollHeight = DOC_HEIGHT;
  scrollTop = 0;
  constructor() {
    super();
    this.top = PANE_TOP;
  }
}

const { BlockWindowController } = await import(
  "../src/components/assistant-ui/block-window-controller.ts"
);

type Controller = InstanceType<typeof BlockWindowController>;

/**
 * A pane holding one document of equal-height blocks, driven a frame at a time.
 *
 * `spacerErrorPx` is the knob that makes the scroll compensation testable: it mis-sizes the
 * spacer on purpose so there is a residual to correct. The design says there should never be one.
 */
function harness(spacerErrorPx = 0) {
  const pane = new FakePane();
  const origin = new FakeElement();
  const slots = new Map<number, FakeElement>();
  const notified: number[] = [];
  const controller: Controller = new BlockWindowController();

  controller.spacerRef(origin as unknown as HTMLElement);
  for (let index = 0; index < BLOCK_COUNT; index += 1) {
    const at = index;
    controller.subscribeBlock(at, () => notified.push(at));
  }
  const detach = controller.attach(pane as unknown as HTMLElement);

  function layout(): void {
    const shift = controller.windowStart() > 0 ? spacerErrorPx : 0;
    origin.top = PANE_TOP - pane.scrollTop;
    for (const [index, slot] of slots) {
      const marker = slot.firstElementChild;
      if (marker) {
        marker.top = PANE_TOP + index * BLOCK_HEIGHT - pane.scrollTop + shift;
      }
    }
    pane.scrollHeight = DOC_HEIGHT + shift;
  }

  /** What React does with the window's answer: mount the suffix, unmount what fell out. */
  function commit(): void {
    const start = controller.windowStart();
    for (const index of [...slots.keys()]) {
      if (index < start) {
        controller.markerRef(index)(null);
        slots.delete(index);
      }
    }
    for (let index = Math.max(1, start); index < BLOCK_COUNT; index += 1) {
      if (slots.has(index)) {
        continue;
      }
      const slot = new FakeElement();
      if (index !== EMPTY_BLOCK) {
        slot.children.push(new FakeElement());
      }
      slots.set(index, slot);
      controller.markerRef(index)(slot as unknown as HTMLElement);
    }
    layout();
  }

  /** One browser frame: the commit is measured, the window moves, the move is committed. */
  function frame(): void {
    for (const observer of [...mutationObservers]) {
      observer.callback();
    }
    commit();
    controller.settleAfterCommit();
  }

  function scrollTo(position: number): void {
    pane.scrollTop = position;
    layout();
  }

  commit();
  return {
    controller,
    pane,
    origin,
    slots,
    notified,
    frame,
    commit,
    scrollTo,
    detach,
    mounted: () => [...slots.keys()].sort((a, b) => a - b),
  };
}

test("a pane at the top mounts the whole document", () => {
  const h = harness();
  h.frame();
  assert.equal(h.controller.windowStart(), 0);
  assert.equal(h.controller.spacerHeight(), 0);
  assert.equal(h.mounted().length, BLOCK_COUNT - 1);
  h.detach();
});

test("scrolled to the bottom, the spacer is exactly the height of the blocks it replaced", () => {
  const h = harness();
  h.frame();
  h.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  h.notified.length = 0;
  h.frame();

  const start = h.controller.windowStart();
  assert.equal(start, 22);
  // Exactly the sum of the dropped blocks' heights, in the pane's own scroll coordinates.
  assert.equal(h.controller.spacerHeight(), start * BLOCK_HEIGHT);
  assert.deepEqual(
    h.mounted(),
    Array.from({ length: BLOCK_COUNT - start }, (_, i) => start + i),
  );

  // Every block that fell out was told, and no block that did not fall out was.
  assert.deepEqual(
    [...new Set(h.notified)].sort((a, b) => a - b),
    Array.from({ length: start }, (_, i) => i),
  );

  // The spacer being exact is what makes the compensation dead code, so assert the 0 rather than
  // trusting the compensation to hide a mistake.
  assert.equal(h.controller.lastResidualPx, 0);
  assert.equal(h.pane.scrollTop, DOC_HEIGHT - PANE_HEIGHT);
  h.detach();
});

test("scrolling back up leaves the reader where they scrolled to", () => {
  // The regression test for the scroll INVERSION. The compensation subtracts the anchor's
  // position after the commit from its position before, so the anchor has to exist in BOTH
  // states. Moving the window BACKWARD, which is what a reader scrolling up causes, the new
  // start is not mounted yet: anchoring on it read the SPACER before and the newly mounted
  // BLOCK after, two different elements, and produced a residual the size of the spacer. In a
  // real browser that wrote 8,271px into a pane a reader had moved 1,440px in, which put them
  // at the bottom, which re-armed the pane's autoscroll and pinned them there for the rest of
  // the generation.
  //
  // The reader has to land on a NON-ZERO window start for this to be the case under test: a
  // scroll back to the very top puts the start at 0, block 0 is never given a marker, and both
  // reads fall back to the spacer and agree by accident.
  const h = harness();
  h.frame();
  h.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  h.frame();
  const forwardStart = h.controller.windowStart();

  h.scrollTo(2600);
  h.frame();
  const backStart = h.controller.windowStart();
  const scrollTop = h.pane.scrollTop;
  const residual = h.controller.lastResidualPx;
  h.detach();

  assert.equal(forwardStart, 22, "the window must have moved forward first");
  assert.ok(
    backStart > 0 && backStart < forwardStart,
    `the window must come back to a non-zero start, got ${backStart}`,
  );
  assert.equal(scrollTop, 2600, "the reader was moved by the compensation");
  assert.equal(residual, 0);
});

test("the correction is not switched off on the way back", () => {
  // The second half of the fix. The gate on applying a correction used to be "the lowest mounted
  // index has reached the anchor", which only describes a window moving FORWARD; a window moving
  // backward mounts blocks BELOW the anchor, so on a scroll-back that test never came true and
  // the correction was dropped every time. `lastResidualPx` is written every time the correction
  // runs, so a stale value here means it did not run at all.
  const h = harness(40);
  h.frame();
  h.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  h.frame();
  const forwardResidual = h.controller.lastResidualPx;

  h.scrollTo(2600);
  h.frame();
  const backStart = h.controller.windowStart();
  const backResidual = h.controller.lastResidualPx;
  h.detach();

  assert.notEqual(
    forwardResidual,
    0,
    "the harness must actually mis-size the spacer, or this proves nothing",
  );
  assert.ok(backStart > 0 && backStart < 22, `start is ${backStart}`);
  assert.equal(
    backResidual,
    0,
    "the correction did not run on the way back: lastResidualPx is still the forward one",
  );
});

test("what stays mounted is bounded by the retained band, not by the document", () => {
  const h = harness();
  h.frame();
  h.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  h.frame();
  const mountedPx = h.mounted().length * BLOCK_HEIGHT;
  assert.ok(mountedPx <= 1900, `mounted ${mountedPx}px`);
  // 18 of 39, i.e. the pane holds less than half of a 4000px document and a proportionally
  // smaller share of a longer one.
  assert.equal(h.mounted().length, 18);
  h.detach();
});

test("scrolling back up remounts only what re-enters, and keeps what was already there", () => {
  const h = harness();
  h.frame();
  h.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  h.frame();
  assert.equal(h.controller.windowStart(), 22);
  const before = new Map(h.slots);

  h.scrollTo(3000);
  h.notified.length = 0;
  h.frame();

  assert.equal(h.controller.windowStart(), 14);
  assert.equal(h.controller.spacerHeight(), 1400);
  // The retained half: the very same slot objects, i.e. React was never asked to remount them.
  for (let index = 22; index < BLOCK_COUNT; index += 1) {
    assert.equal(
      h.slots.get(index),
      before.get(index),
      `block ${index} was remounted for no reason`,
    );
  }
  // And only the blocks that re-entered were told anything.
  assert.deepEqual(
    [...new Set(h.notified)].sort((a, b) => a - b),
    [14, 15, 16, 17, 18, 19, 20, 21],
  );
  h.detach();
});

test("an empty block has no marker and is simply never a window start", () => {
  const h = harness();
  h.frame();
  h.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  h.frame();
  h.scrollTo(EMPTY_BLOCK * BLOCK_HEIGHT + 1536 + 50);
  h.frame();
  assert.notEqual(
    h.controller.windowStart(),
    EMPTY_BLOCK,
    "a block that renders nothing has no measured height to freeze",
  );
  h.detach();
});

test("a mis-sized spacer is corrected explicitly, and only when the reader is not pinned", () => {
  const wrong = harness(9);
  wrong.frame();
  // Away from the bottom, so the pane is not pinned and the reader's position is the thing that
  // has to be preserved.
  wrong.scrollTo(2600);
  wrong.frame();
  assert.ok(wrong.controller.windowStart() > 0);
  assert.equal(wrong.controller.lastResidualPx, 9);
  assert.equal(
    wrong.pane.scrollTop,
    2609,
    "the reader's anchor moved 9px and was put back",
  );
  wrong.detach();

  // Pinned to the bottom, the autoscroll owns the scroll position: writing to it here would read
  // to the autoscroll's own handler as the reader scrolling up, and would detach it.
  const pinned = harness(9);
  pinned.frame();
  pinned.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  pinned.frame();
  assert.equal(pinned.controller.lastResidualPx, 9);
  assert.equal(pinned.pane.scrollTop, DOC_HEIGHT - PANE_HEIGHT);
  pinned.detach();
});

test("a width change throws every frozen height away", () => {
  const h = harness();
  h.frame();
  h.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  h.frame();
  assert.equal(h.controller.windowStart(), 22);

  h.pane.clientHeight = 512;
  for (const observer of [...resizeObservers]) {
    observer.callback();
  }
  assert.equal(
    h.controller.windowStart(),
    22,
    "a height change is the pane's own cap animating and means nothing",
  );

  h.pane.clientWidth = 420;
  for (const observer of [...resizeObservers]) {
    observer.callback();
  }
  assert.equal(h.controller.windowStart(), 0);
  assert.equal(h.controller.spacerHeight(), 0);

  // And it really does remount and re-measure rather than leaving a stale map behind.
  h.commit();
  assert.equal(h.mounted().length, BLOCK_COUNT - 1);
  h.detach();
});

test("a re-parse behind the live edge throws every frozen height away", () => {
  const h = harness();
  h.frame();
  h.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  h.frame();
  assert.equal(h.controller.windowStart(), 22);

  for (let index = 0; index < BLOCK_COUNT; index += 1) {
    h.controller.reportContent(index, `block ${index}`);
  }
  h.controller.reportContent(BLOCK_COUNT - 1, "block 39 with another token");
  assert.equal(
    h.controller.windowStart(),
    22,
    "the newest block growing is not a re-parse",
  );

  // A late GFM footnote definition collapsing earlier blocks looks exactly like this.
  h.controller.reportContent(4, "blocks 4 through 9, merged");
  assert.equal(h.controller.windowStart(), 0);
  assert.equal(h.controller.spacerHeight(), 0);
  h.detach();
});

test("the observer watches the spacer and the top of the window, and moves with it", () => {
  const h = harness();
  h.frame();
  const [observer] = intersectionObservers;
  assert.ok(observer, "no IntersectionObserver was created");
  assert.ok(observer.targets.has(h.origin), "the spacer must be watched");

  h.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  h.frame();
  const start = h.controller.windowStart();
  const watched = [...observer.targets];
  assert.ok(watched.includes(h.origin));
  for (const index of [start, start + 1, start + 2]) {
    assert.ok(
      watched.includes(h.slots.get(index)?.firstElementChild),
      `the observer must watch block ${index} at the top of the window`,
    );
  }
  // Not every mounted block: the point is a handful at the edge, not another per-block cost.
  assert.ok(watched.length <= 4, `watching ${watched.length} elements`);
  h.detach();
});

test("scrolling with no DOM change still moves the window", () => {
  const h = harness();
  h.frame();
  // No mutation callback at all: only the intersection observer reports this, because scroll
  // events are deliberately not used.
  h.scrollTo(DOC_HEIGHT - PANE_HEIGHT);
  for (const observer of [...intersectionObservers]) {
    observer.callback();
  }
  assert.equal(h.controller.windowStart(), 22);
  h.detach();
});

test("a block's marker ref is the same function on every render", () => {
  // React detaches and re-attaches a callback ref whose identity changed, on every render. A
  // fresh function per render would therefore unregister and re-register every marker in the
  // window on every token, and the measurement would be reading a map that had just been emptied.
  const h = harness();
  assert.equal(h.controller.markerRef(5), h.controller.markerRef(5));
  assert.notEqual(h.controller.markerRef(5), h.controller.markerRef(6));
  h.detach();
});

test("detaching stops the observers", () => {
  const h = harness();
  h.frame();
  const before = mutationObservers.length;
  h.detach();
  assert.equal(mutationObservers.length, before - 1);
});
