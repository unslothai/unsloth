// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The wheel zooms the canvas over cards, notes and groups; it scrolls a box only while that box can move.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { wheelScrollsInside } from "../src/features/flows/editor/wheel-scroll.ts";

type B = {
  scrollTop: number; scrollLeft: number; scrollHeight: number; scrollWidth: number;
  clientHeight: number; clientWidth: number; parentElement: B | null; overflowY: string; overflowX: string;
};
const box = (o: Partial<B>, parent: B | null): B => ({
  scrollTop: 0, scrollLeft: 0, scrollHeight: 100, scrollWidth: 100, clientHeight: 100, clientWidth: 100,
  overflowY: "visible", overflowX: "visible", parentElement: parent, ...o,
});
const styleOf = (el: object) => {
  const b = el as B;
  return { overflowY: b.overflowY, overflowX: b.overflowX };
};

test("over a card or group with nothing to scroll, the wheel zooms", () => {
  const canvas = box({}, null);
  const card = box({ overflowY: "clip", scrollHeight: 400 }, canvas);
  const text = box({}, card);
  assert.equal(wheelScrollsInside(text, canvas, 0, -50, false, styleOf), false);
});

test("a note with more text than fits scrolls until its end, then the wheel zooms", () => {
  const canvas = box({}, null);
  const note = box({ overflowY: "auto", scrollHeight: 300, clientHeight: 100 }, canvas);
  const line = box({}, note);
  assert.equal(wheelScrollsInside(line, canvas, 0, 40, false, styleOf), true);
  note.scrollTop = 200; // at the bottom
  assert.equal(wheelScrollsInside(line, canvas, 0, 40, false, styleOf), false);
  assert.equal(wheelScrollsInside(line, canvas, 0, -40, false, styleOf), true);
  // A short note never steals the wheel.
  const short = box({ overflowY: "auto" }, canvas);
  assert.equal(wheelScrollsInside(short, canvas, 0, 40, false, styleOf), false);
});

test("a pinch zooms even over a box that could scroll", () => {
  const canvas = box({}, null);
  const note = box({ overflowY: "auto", scrollHeight: 300 }, canvas);
  assert.equal(wheelScrollsInside(note, canvas, 0, 40, true, styleOf), false);
});

test("the canvas decides in the capture phase, and React Flow's nowheel class no longer blocks zoom", () => {
  const canvas = readSrc("features/flows/editor/canvas.tsx");
  assert.match(canvas, /onWheelCapture=\{\(event\) => \{[\s\S]{0,300}wheelScrollsInside\([\s\S]{0,200}event\.stopPropagation\(\)/);
  assert.match(canvas, /noWheelClassName="flows-wheel-unused"/);
});

test("a note's text is laid out in the note body, so the body is the box the wheel scrolls", () => {
  const frames = readSrc("features/flows/editor/frame-nodes.tsx");
  // Plain markdown takes no pointer; as its own h-full scroller it could never be wheeled.
  assert.match(frames, /<MarkdownPreview\s*markdown=\{note\.text\}\s*plain=\{true\}\s*className="h-auto overflow-visible"/);
  assert.match(frames, /<BaseNodeContent className="[^"]*h-full overflow-y-auto/);
});
