// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  THREAD_GUTTER_CHANGE_EVENT,
  threadGutterShift,
} from "../src/components/assistant-ui/thread-gutter.ts";
import { readSrc } from "./helpers/kit.ts";

const SHIFT_RULE =
  /\.aui-thread-viewport\[data-gutter-shift\] \{\s*scrollbar-gutter: stable;\s*padding-left: calc\(var\(--spacing\) \* 5 \+ var\(--thread-gutter-shift\)\);/;
const FULL_VIEW_SHIFT_RULE =
  /\.chat-full-view-dock \.aui-thread-viewport\[data-gutter-shift\] \{\s*padding-left: calc\(1rem \* var\(--ui-space-scale, 1\) \+ var\(--thread-gutter-shift\)\);/;
const WATCH_CALL = /watchThreadGutter\(viewportEl\)/;
const DOCK_LISTENER =
  /viewport\.addEventListener\(THREAD_GUTTER_CHANGE_EVENT, apply\)/;

test("a misplaced both-edges gutter shifts the column by one scrollbar", () => {
  assert.equal(threadGutterShift(13, true, false), 13);
});

test("engines that reserve the left gutter, overlay scroll bars and RTL are left alone", () => {
  assert.equal(threadGutterShift(13, false, false), 0);
  assert.equal(threadGutterShift(0, true, false), 0);
  assert.equal(threadGutterShift(13, true, true), 0);
});

test("the shift swaps to a one-sided gutter and pads the left, in both viewport paddings", () => {
  const css = readSrc("index.css");
  assert.match(css, SHIFT_RULE);
  assert.match(css, FULL_VIEW_SHIFT_RULE);
});

test("the composer dock re-measures when the shift moves the column", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, WATCH_CALL);
  assert.match(thread, DOCK_LISTENER);
  assert.equal(THREAD_GUTTER_CHANGE_EVENT, "unsloth:thread-gutter-change");
});
