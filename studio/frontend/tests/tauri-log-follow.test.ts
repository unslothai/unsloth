// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  STICK_THRESHOLD_PX,
  isFollowingTail,
} from "../src/components/tauri/log-follow.ts";

import { readSrcAsync } from "./helpers/kit.ts";

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
  // Fractional line heights and zoom land just short of the end; that is not a scroll-up.
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
  // Hidden <details> content reads all metrics as 0, which must not latch the follow off.
  assert.equal(
    isFollowingTail({ scrollHeight: 0, scrollTop: 0, clientHeight: 0 }),
    true,
  );
});

test("LogDetails drives its scrolling through the shared predicate", async () => {
  const source = await readSrcAsync("components/tauri/log-details.tsx");

  assert.match(source, /isFollowingTail\(log\)/);
  assert.match(source, /onScroll=\{handleScroll\}/);
  assert.match(source, /onToggle=\{handleToggle\}/);
  // A passive effect paints new lines at the old offset first, which judders.
  assert.match(source, /useLayoutEffect/);
  // A render per scroll event is too costly for a streaming log, so no useState.
  assert.doesNotMatch(source, /useState/);
});
