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
