// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pacing of a full mount and the anchor arithmetic shared by every window commit. Which rows are
// mounted is tested in thread-window.test.ts; the .tsx glue in progressive-mount-glue.test.ts.

import assert from "node:assert/strict";
import test from "node:test";

import {
  CHUNK_MESSAGES,
  anchorCorrection,
  FAR_MOUNT_CHUNK,
  nextMountChunk,
  watchProgressiveScroll,
} from "../src/components/assistant-ui/progressive-mount-controller.ts";
import { type RowWindow, planFill } from "../src/components/assistant-ui/thread-window.ts";

test("mounting waits for wheel momentum and thumb release, without pausing on its own corrections", () => {
  const viewport = new EventTarget();
  let time = 1000;
  const activity = watchProgressiveScroll(viewport, () => time);
  const send = (type: string) => viewport.dispatchEvent(new Event(type));
  assert.equal(activity.blocked(false), true);
  time += 301;
  send("scroll");
  assert.equal(activity.blocked(false), false);
  send("wheel");
  assert.equal(activity.blocked(false), true);
  time += 250;
  send("scroll");
  time += 250;
  assert.equal(activity.blocked(false), true);
  time += 51;
  assert.equal(activity.blocked(false), false);
  send("scroll");
  assert.equal(activity.blocked(false), false);
  assert.equal(activity.blocked(true), true);
  time += 1000;
  assert.equal(activity.blocked(true), true);
  assert.equal(activity.blocked(false), true);
  time += 301;
  assert.equal(activity.blocked(false), false);
  activity.dispose();
  send("wheel");
  assert.equal(activity.blocked(false), false);
});

test("scroll keys, touch momentum, and composer input outside the viewport pause mounting", () => {
  const document = new EventTarget();
  const viewport = Object.assign(new EventTarget(), { ownerDocument: document });
  let time = 1000;
  const activity = watchProgressiveScroll(viewport, () => time);
  time += 301;
  viewport.dispatchEvent(Object.assign(new Event("keydown"), { key: "a" }));
  assert.equal(activity.blocked(false), false);
  viewport.dispatchEvent(Object.assign(new Event("keydown"), { key: "PageUp" }));
  assert.equal(activity.blocked(false), true);
  time += 301;
  viewport.dispatchEvent(new Event("touchmove"));
  assert.equal(activity.blocked(false), true);
  time += 301;
  document.dispatchEvent(new Event("beforeinput"));
  assert.equal(activity.blocked(false), true);
  time += 301;
  document.dispatchEvent(new Event("compositionupdate"));
  assert.equal(activity.blocked(false), true);
  time += 301;
  document.dispatchEvent(new Event("pointerdown"));
  assert.equal(activity.blocked(false), true);
  activity.dispose();
  time += 301;
  document.dispatchEvent(new Event("beforeinput"));
  assert.equal(activity.blocked(false), false);
});

test("a stream's own autoscroll does not hold a full mount off, only the reader's momentum does", () => {
  const viewport = new EventTarget();
  let time = 1000;
  const activity = watchProgressiveScroll(viewport, () => time);
  const send = (type: string) => viewport.dispatchEvent(new Event(type));
  // Inside the opening pause, scrolls with no input behind them (the autoscroll) do not renew it.
  for (let at = 0; at < 10; at += 1) {
    time += 100;
    send("scroll");
  }
  assert.equal(activity.blocked(false), false);
  // A fling: momentum scrolls keep the pause going for as long as they run, input or not.
  send("touchmove");
  for (let at = 0; at < 20; at += 1) {
    time += 100;
    send("scroll");
    assert.equal(activity.blocked(false), true);
  }
  // Past the momentum allowance, scrolling alone lets the full mount go on.
  time += 1100;
  send("scroll");
  time += 301;
  assert.equal(activity.blocked(false), false);
});

const sample = (viewportOffset: number, scrollTop: number, maxScrollTop = 1_000_000) => ({
  viewportOffset,
  scrollTop,
  maxScrollTop,
});

test("the correction is the height inserted above, in document space", () => {
  assert.equal(anchorCorrection(sample(-500, 20000), sample(11500, 20000)), 12000);
});

test("the reader's own scroll is subtracted, not skipped", () => {
  // Document space nets reader scroll against insertion above.
  assert.equal(anchorCorrection(sample(-500, 20000), sample(8500, 23000)), 12000);
  assert.equal(anchorCorrection(sample(-500, 20000), sample(-3500, 23000)), null);
});

test("a frame that inserted nothing is a no-op whatever the reader did", () => {
  for (const scrolled of [0, -900, 4000]) {
    assert.equal(
      anchorCorrection(sample(-500, 20000), sample(-500 - scrolled, 20000 + scrolled)),
      null,
      `no insertion should mean no correction (reader moved ${scrolled}px)`,
    );
  }
});

test("sub-pixel movement is never acted on", () => {
  assert.equal(anchorCorrection(sample(-500.0, 20000), sample(-499.4, 20000)), null);
});

test("a shrink the browser already clamped is not corrected twice", () => {
  // With anchoring off the browser clamps scrollTop on shrink; apply only the unabsorbed part.
  assert.equal(
    anchorCorrection(sample(200, 9000, 9100), sample(100, 8600, 8600)),
    -100,
  );
  assert.equal(
    anchorCorrection(sample(200, 9000, 50_000), sample(-300, 9000, 49_500)),
    -500,
  );
});

test("WebKitGTK steps stay small near the reader and ramp to FAR_MOUNT_CHUNK past the window reach", () => {
  const near = 2;
  const reach = 2000;
  // Unknown geometry or a frontier within reach (or below the viewport top) keeps the small step.
  assert.equal(nextMountChunk(near, null, near, reach), near);
  assert.equal(nextMountChunk(CHUNK_MESSAGES, 0, near, reach), near);
  assert.equal(nextMountChunk(CHUNK_MESSAGES, -500, near, reach), near);
  assert.equal(nextMountChunk(CHUNK_MESSAGES, reach, near, reach), near);
  assert.equal(nextMountChunk(CHUNK_MESSAGES, Number.NaN, near, reach), near);
  // Past it the first step of a window (previous 0) stays small, so the tail paints first; then it
  // grows to the cap and holds.
  const steps: number[] = [];
  let chunk = 0;
  for (let i = 0; i < 6; i += 1) {
    chunk = nextMountChunk(chunk, reach + 1, near, reach);
    steps.push(chunk);
  }
  assert.deepEqual(steps, [near, 8, FAR_MOUNT_CHUNK, FAR_MOUNT_CHUNK, FAR_MOUNT_CHUNK, FAR_MOUNT_CHUNK]);
  assert.ok(FAR_MOUNT_CHUNK <= 16, "larger far steps made ~1.4 s frames on CPU-painted WebKit");
  // A step after near ones ramps from the near size.
  assert.equal(nextMountChunk(near, reach + 1, near, reach), 8);
  // Coming back within reach (the reader scrolled up to the frontier) drops straight to the small step.
  assert.equal(nextMountChunk(CHUNK_MESSAGES, 100, near, reach), near);
  // The near step already being the cap (other engines) never changes it.
  assert.equal(nextMountChunk(CHUNK_MESSAGES, 10_000, CHUNK_MESSAGES, reach), CHUNK_MESSAGES);
  assert.equal(nextMountChunk(CHUNK_MESSAGES, null, CHUNK_MESSAGES, reach), CHUNK_MESSAGES);
  // A 300-message thread past the reach is fully mounted in about 20 commits instead of about 142.
  let current: RowWindow | null = { start: 298, end: 300 };
  let commits = 0;
  chunk = 0;
  while (current != null) {
    chunk = nextMountChunk(chunk, reach + 1, near, reach);
    current = planFill(current, 300, chunk);
    commits += 1;
  }
  assert.ok(commits <= 24, `took ${commits} commits`);
});
