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

test("a monitor filling the height still leaves the stack room", () => {
  const frame: MonitorFrame = {
    left: W - 272,
    top: 16,
    right: W - 16,
    bottom: H - 16,
  };
  assert.ok(stackBottomInset(frame, W, H) <= H - 120);
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
  assert.ok(both.bottom < unioned.bottom, "and it must lift less, not more");
});

test("an empty list behaves exactly like nothing published", () => {
  assert.deepEqual(stackGeometry([], W, H), stackGeometry(null, W, H));
});

test("one box in a list matches passing it on its own", () => {
  const frame = corner(300);
  assert.deepEqual(stackGeometry([frame], W, H), stackGeometry(frame, W, H));
});

// The two composer layouts, which are what this gate exists for. Boxes taken
// from a 1280x830 window: the docked one has to be dodged, or the card covers
// Send, which below a 1584px viewport it does. The welcome one must not be,
// because it sits high on the page and lifting over it stranded the banners in
// the middle of the screen with the corner underneath them empty.
const CHAT_W = 1280;
const CHAT_H = 830;

test("a docked composer is dodged, so the card cannot cover Send", () => {
  const docked = { left: 412, top: 664, right: 1148, bottom: 814 };
  const inset = stackBottomInset(docked, CHAT_W, CHAT_H);
  assert.ok(inset > 16, "it reaches the stack's strip, so the stack lifts");
  assert.ok(
    inset >= CHAT_H - docked.top,
    "and lifts clear of it, not part way",
  );
  // Still the bottom of the screen, which is the point: above the composer,
  // not adrift in the middle.
  assert.ok(inset < CHAT_H / 2);
});

test("the welcome composer is left alone, and the stack stays in the corner", () => {
  const welcome = { left: 412, top: 435, right: 1148, bottom: 660 };
  assert.equal(stackBottomInset(welcome, CHAT_W, CHAT_H), 16);
});

// Same rule, applied to the other publisher: a monitor dragged up the screen
// leaves the corner free, so the stack belongs in it.
test("a monitor away from the corner no longer lifts the stack", () => {
  const middle = { left: 996, top: 300, right: 1264, bottom: 560 };
  const geometry = stackGeometry(middle, CHAT_W, CHAT_H);
  assert.equal(geometry.bottom, 16);
  // It is still in the column, so the stack is capped short of it instead.
  assert.ok(geometry.maxHeight < CHAT_H - 16 - 16);
  assert.ok(geometry.maxHeight <= CHAT_H - 16 - middle.bottom);
});
