// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type MonitorFrame,
  stackBottomInset,
  stackGeometry,
} from "../src/features/settings/stores/monitor-frame-store.ts";

const W = 1440;
const H = 900;

/** The Live monitor where it opens by default: bottom-right, w-64, inset-4. */
function corner(height = 300): MonitorFrame {
  return {
    left: W - 16 - 256,
    top: H - 16 - height,
    right: W - 16,
    bottom: H - 16,
  };
}

test("with no monitor open the stack keeps its own inset", () => {
  assert.equal(stackBottomInset(null, W, H), 16);
});

// The reported overlap: both default to the same corner.
test("a monitor in its default corner lifts the stack clear of it", () => {
  const frame = corner(300);
  const inset = stackBottomInset(frame, W, H);
  // The stack's bottom edge now sits above the monitor's top edge.
  assert.ok(inset > 16, "the stack must move");
  assert.ok(H - inset <= frame.top, "no vertical overlap remains");
});

test("a monitor dragged to the left is not dodged", () => {
  const frame: MonitorFrame = {
    left: 16,
    top: H - 316,
    right: 272,
    bottom: H - 16,
  };
  assert.equal(stackBottomInset(frame, W, H), 16);
});

test("a monitor dragged to the top right is not dodged", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 16,
    right: W - 16,
    bottom: 316,
  };
  assert.equal(stackBottomInset(frame, W, H), 16);
});

// The update banners are max-w-[448px], wider than the download panel, so the
// column has to be measured from them.
test("a monitor beside the download panel but under a banner is dodged", () => {
  const frame: MonitorFrame = {
    left: W - 16 - 430,
    top: H - 316,
    right: W - 16 - 410,
    bottom: H - 16,
  };
  assert.ok(stackBottomInset(frame, W, H) > 16);
});

// Lifting the stack without shortening it pushes its top off the screen.
test("lifting the stack shortens it by the same amount", () => {
  assert.equal(stackGeometry(null, W, H).maxHeight, H - 32);
  const lifted = stackGeometry(corner(300), W, H);
  assert.ok(lifted.bottom > 16);
  assert.equal(lifted.maxHeight, H - lifted.bottom - 16);
});

// A monitor parked high is not lifted over, since the room is beneath it, but
// the stack grows upwards and would still run into it.
test("a monitor high in the column caps the stack instead of lifting it", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 16,
    right: W - 16,
    bottom: 316,
  };
  const geometry = stackGeometry(frame, W, H);
  // Nothing to lift over: the free space is below it.
  assert.equal(geometry.bottom, 16);
  // The stack's top edge stays clear of the monitor's bottom edge.
  assert.ok(H - geometry.bottom - geometry.maxHeight >= frame.bottom);
  assert.ok(geometry.maxHeight < H - 32, "it must actually be capped");
});

test("a monitor high but outside the column does not cap the stack", () => {
  const frame: MonitorFrame = { left: 16, top: 16, right: 272, bottom: 316 };
  assert.equal(stackGeometry(frame, W, H).maxHeight, H - 32);
});

// A monitor filling almost the whole column would otherwise leave no stack.
test("the cap never shrinks the stack below its floor", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 0,
    right: W - 16,
    bottom: H / 2 - 1,
  };
  assert.ok(stackGeometry(frame, W, H).maxHeight >= 120);
});

// Clamping the lift to the stack's floor was worse than not lifting: it put
// the stack across the monitor's own top edge, which is where its Close and
// collapse controls are. The chat UI suite maximises the monitor and then
// clicks Close, and the card swallowed the click.
test("a monitor too tall to lift over is left in the corner", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 16,
    right: W - 16,
    bottom: H - 16,
  };
  const inset = stackBottomInset(frame, W, H);
  assert.equal(inset, 16, "no lift can clear it, so do not half-lift");
  // The monitor's own controls sit just inside its top edge; the stack must
  // stay well below them.
  assert.ok(H - inset > frame.top + 64, "the stack stays off the top bar");
});

test("a monitor resized to fill the viewport is left in the corner", () => {
  const frame: MonitorFrame = { left: 16, top: 16, right: W - 16, bottom: H - 16 };
  assert.equal(stackBottomInset(frame, W, H), 16);
});

// The lift is dropped only when it cannot clear; one that fits still applies.
test("a tall monitor that can still be cleared is lifted over", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 200,
    right: W - 16,
    bottom: H - 16,
  };
  const inset = stackBottomInset(frame, W, H);
  assert.ok(inset > 16, "the stack must move");
  assert.ok(H - inset <= frame.top, "no vertical overlap remains");
  assert.ok(H - inset - 16 >= 120, "and it keeps its floor");
});

// The union was the trap. A tall monitor and the wide docked composer share
// almost no area, so the rectangle around the pair covers most of the viewport;
// reading that as one obstacle lifted the stack to the top of the screen and
// dropped it back onto the monitor it was meant to dodge, which the chat UI
// suite caught as the card swallowing the monitor's Close button.
test("two obstacles are folded one at a time, not as their bounding box", () => {
  // The monitor dragged up the column, as the chat UI suite does before it
  // clicks Close: too high to be lifted over, so on its own it asks for
  // nothing. The composer, docked, asks for a modest lift.
  const monitor = { left: W - 16 - 256, top: 40, right: W - 16, bottom: 340 };
  const composer = { left: 300, top: H - 120, right: W - 340, bottom: H - 40 };
  const both = stackGeometry([monitor, composer], W, H);
  const monitorOnly = stackGeometry(monitor, W, H);
  const composerOnly = stackGeometry(composer, W, H);
  assert.equal(
    both.bottom,
    Math.max(monitorOnly.bottom, composerOnly.bottom),
    "the stack takes the largest lift either one asks for",
  );
  // The union's own answer, which is what went wrong.
  const unioned = stackGeometry(
    {
      left: Math.min(monitor.left, composer.left),
      top: Math.min(monitor.top, composer.top),
      right: Math.max(monitor.right, composer.right),
      bottom: Math.max(monitor.bottom, composer.bottom),
    },
    W,
    H,
  );
  assert.notEqual(
    both.bottom,
    unioned.bottom,
    "folding must not agree with the bounding box, or nothing was fixed",
  );
  // The union is wrong in whichever direction the clamp happens to send it:
  // it used to lift to the cap and land on the monitor, and now that a box too
  // tall to clear is left alone it drops the composer's dodge instead, putting
  // the card back over the Send button. Folding just gives each box its own.
  assert.equal(
    both.bottom,
    composerOnly.bottom,
    "the composer still gets the lift it asked for",
  );
});

test("an empty list behaves exactly like nothing published", () => {
  assert.deepEqual(stackGeometry([], W, H), stackGeometry(null, W, H));
});

test("one box in a list matches passing it on its own", () => {
  const frame = corner(300);
  assert.deepEqual(stackGeometry([frame], W, H), stackGeometry(frame, W, H));
});
