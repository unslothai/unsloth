// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  THREAD_GUTTER_CHANGE_EVENT,
  THREAD_GUTTER_SHIFT_ATTRIBUTE,
  syncThreadGutter,
  threadGutterShift,
  watchThreadGutter,
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

const LAYOUT_WATCH =
  /useLayoutEffect\(\(\) => \{\s*if \(!viewportEl\) return;\s*return watchThreadGutter\(viewportEl\);/;
const DOCK_SYNCS_FIRST =
  /const apply = \(\) => \{[^}]*?syncThreadGutter\(viewport\);[\s\S]*?const footer = childWithClass/;

function fakeViewport() {
  const attrs = new Set<string>();
  const props = new Map<string, string>();
  const events: string[] = [];
  const box = { left: 0, width: 100 };
  const doc = {
    createElement: () => ({
      setAttribute: () => undefined,
      style: {},
      append: () => undefined,
      remove: () => undefined,
      offsetWidth: 100,
      clientWidth: 85,
      getBoundingClientRect: () => box,
    }),
    body: { append: () => undefined },
  };
  const viewport = {
    offsetWidth: 600,
    clientWidth: 600,
    ownerDocument: doc,
    style: {
      setProperty: (k: string, v: string) => props.set(k, v),
      removeProperty: (k: string) => props.delete(k),
    },
    toggleAttribute: (name: string, on: boolean) =>
      on ? attrs.add(name) : attrs.delete(name),
    removeAttribute: (name: string) => attrs.delete(name),
    hasAttribute: (name: string) => attrs.has(name),
    dispatchEvent: (event: Event) => events.push(event.type),
  };
  return { viewport, attrs, events };
}

test("a caller can bring the shift up to date before its own observer would", () => {
  const g = globalThis as Record<string, unknown>;
  const saved = [g.ResizeObserver, g.getComputedStyle];
  g.ResizeObserver = class {
    observe = () => undefined;
    disconnect = () => undefined;
  };
  g.getComputedStyle = () => ({
    borderLeftWidth: "0px",
    borderRightWidth: "0px",
    direction: "ltr",
  });
  try {
    const { viewport, attrs, events } = fakeViewport();
    const el = viewport as unknown as HTMLElement;
    const stop = watchThreadGutter(el);
    assert.equal(attrs.has(THREAD_GUTTER_SHIFT_ATTRIBUTE), false);
    viewport.clientWidth = 585;
    syncThreadGutter(el);
    assert.equal(attrs.has(THREAD_GUTTER_SHIFT_ATTRIBUTE), true);
    assert.deepEqual(events, [THREAD_GUTTER_CHANGE_EVENT]);
    stop();
    viewport.clientWidth = 600;
    syncThreadGutter(el);
    assert.equal(attrs.has(THREAD_GUTTER_SHIFT_ATTRIBUTE), false);
  } finally {
    [g.ResizeObserver, g.getComputedStyle] = saved;
  }
});

test("a remounted viewport is shifted before paint, and the dock syncs the gutter before it measures", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, LAYOUT_WATCH);
  assert.match(thread, DOCK_SYNCS_FIRST);
});
