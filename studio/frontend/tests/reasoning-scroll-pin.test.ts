// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createReasoningScrollPin } from "../src/components/assistant-ui/reasoning-scroll-pin.ts";

test("a mutation burst performs at most one reasoning pin per frame", () => {
  const frames = new Map<number, FrameRequestCallback>();
  let nextHandle = 1;
  let pins = 0;
  const attached = true;
  const scrollPin = createReasoningScrollPin(
    () => attached,
    () => {
      pins += 1;
    },
    (callback) => {
      const handle = nextHandle;
      nextHandle += 1;
      frames.set(handle, callback);
      return handle;
    },
    (handle) => frames.delete(handle),
  );

  for (let mutation = 0; mutation < 100; mutation += 1) {
    scrollPin.schedule();
  }
  assert.equal(frames.size, 1);
  assert.equal(pins, 0);

  const first = frames.entries().next().value;
  assert.ok(first);
  frames.delete(first[0]);
  first[1](0);
  assert.equal(pins, 1);

  for (let mutation = 0; mutation < 100; mutation += 1) {
    scrollPin.schedule();
  }
  assert.equal(frames.size, 1, "the next frame can schedule one new pin");
});

test("detach, reattach, and cleanup are checked at the scheduled frame", () => {
  const frames = new Map<number, FrameRequestCallback>();
  let nextHandle = 1;
  let pins = 0;
  let attached = true;
  const scrollPin = createReasoningScrollPin(
    () => attached,
    () => {
      pins += 1;
    },
    (callback) => {
      const handle = nextHandle;
      nextHandle += 1;
      frames.set(handle, callback);
      return handle;
    },
    (handle) => frames.delete(handle),
  );

  scrollPin.schedule();
  attached = false;
  const detachedFrame = frames.entries().next().value;
  assert.ok(detachedFrame);
  frames.delete(detachedFrame[0]);
  detachedFrame[1](0);
  assert.equal(pins, 0, "scrolling up before paint detaches the pin");

  attached = true;
  scrollPin.schedule();
  const reattachedFrame = frames.entries().next().value;
  assert.ok(reattachedFrame);
  frames.delete(reattachedFrame[0]);
  reattachedFrame[1](0);
  assert.equal(pins, 1, "returning to the bottom reattaches the pin");

  scrollPin.schedule();
  assert.equal(frames.size, 1);
  scrollPin.cancel();
  assert.equal(frames.size, 0, "cleanup cancels the pending layout operation");
});
