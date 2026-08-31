// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The install and update logs stream while the user is stuck watching them, which is
// exactly when someone scrolls up to read a line that went past. Following the tail is
// only acceptable if that scroll-up wins until they come back down on their own.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  STICK_THRESHOLD_PX,
  isFollowingTail,
} from "../src/components/tauri/log-follow.ts";

/** A log box 100px tall holding 500px of lines: 400px of travel, bottom at 400. */
const TALL = { scrollHeight: 500, clientHeight: 100 };

test("a log parked at the bottom keeps following", () => {
  assert.equal(isFollowingTail({ ...TALL, scrollTop: 400 }), true);
});

test("a log scrolled up stops following", () => {
  assert.equal(isFollowingTail({ ...TALL, scrollTop: 399 - STICK_THRESHOLD_PX }), false);
  assert.equal(isFollowingTail({ ...TALL, scrollTop: 0 }), false);
});

test("scrolling back down to the bottom resumes the follow", () => {
  assert.equal(isFollowingTail({ ...TALL, scrollTop: 0 }), false);
  assert.equal(isFollowingTail({ ...TALL, scrollTop: 400 }), true);
});

test("a sub-pixel gap at the end still counts as the bottom", () => {
  // Fractional line heights and browser zoom land the offset just short of the end;
  // treating that as a manual scroll-up would stop the follow nobody asked to stop.
  assert.equal(isFollowingTail({ ...TALL, scrollTop: 399.4 }), true);
  assert.equal(isFollowingTail({ ...TALL, scrollTop: 400 - STICK_THRESHOLD_PX }), true);
});

test("a log shorter than its box is always at its own end", () => {
  assert.equal(
    isFollowingTail({ scrollHeight: 40, scrollTop: 0, clientHeight: 100 }),
    true,
  );
});

test("a closed panel reports zeroes and stays armed to follow", () => {
  // Hidden <details> content has no layout, so every metric reads 0. That must not latch
  // the follow off for the lines that arrive before the user opens it.
  assert.equal(
    isFollowingTail({ scrollHeight: 0, scrollTop: 0, clientHeight: 0 }),
    true,
  );
});

test("LogDetails drives its scrolling through the shared predicate", async () => {
  const source = await readFile(
    new URL("../src/components/tauri/log-details.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /isFollowingTail\(log\)/);
  assert.match(source, /onScroll=\{handleScroll\}/);
  assert.match(source, /onToggle=\{handleToggle\}/);
  // A passive effect paints the new lines at the old offset first, which reads as a
  // judder on every append.
  assert.match(source, /useLayoutEffect/);
  // Following state must not go through useState: a render per scroll event is the one
  // cost a streaming log cannot carry.
  assert.doesNotMatch(source, /useState/);
});
