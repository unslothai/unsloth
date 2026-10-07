// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  HYSTERESIS_VIEWPORTS,
  lineIsWindowed,
  OVERSCAN_VIEWPORTS,
  selectLineWindow,
  WINDOW_CAP_LINES,
  type LineWindow,
  type WindowGeometry,
} from "../src/components/assistant-ui/code-fence-window.ts";

import { readSrc } from "./helpers/kit.ts";

const geometry = (over: Partial<WindowGeometry> = {}): WindowGeometry => ({
  lineCount: 20_000,
  lineHeight: 20,
  contentTop: 0,
  viewportTop: 0,
  viewportHeight: 600,
  previous: null,
  ...over,
});

test("a fence at or under the cap is never windowed", () => {
  assert.equal(selectLineWindow(geometry({ lineCount: 1 })), null);
  assert.equal(selectLineWindow(geometry({ lineCount: WINDOW_CAP_LINES })), null);
  assert.equal(selectLineWindow(geometry({ lineCount: WINDOW_CAP_LINES - 1 })), null);
  assert.equal(selectLineWindow(geometry({ lineCount: 0 })), null);
});

test("past the cap the window covers the viewport plus one viewport each way", () => {
  const window = selectLineWindow(geometry({ viewportTop: 10_000 }));
  assert.ok(window, "a 20,000 line fence is windowed");
  const perViewport = 600 / 20;
  assert.equal(window.first, 500 - perViewport * OVERSCAN_VIEWPORTS);
  assert.equal(window.last, 500 + perViewport - 1 + perViewport * OVERSCAN_VIEWPORTS);
});

test("the window is clamped to the fence at both ends", () => {
  const top = selectLineWindow(geometry({ viewportTop: 0 }));
  assert.ok(top);
  assert.equal(top.first, 0, "no negative first line");

  const bottom = selectLineWindow(
    geometry({ viewportTop: 20_000 * 20 - 600 }),
  );
  assert.ok(bottom);
  assert.equal(bottom.last, 19_999, "no line past the end");
});

test("the window is read against the viewport, not the fence, when the fence starts off screen", () => {
  const window = selectLineWindow(geometry({ contentTop: -5_000 }));
  assert.ok(window);
  assert.equal(window.first, 250 - 30);
  assert.equal(window.last, 250 + 30 - 1 + 30);
});

test("an ordinary scroll inside the window does not move it", () => {
  const previous: LineWindow = { first: 470, last: 559 };
  const kept = selectLineWindow(
    geometry({ viewportTop: 10_040, previous }),
  );
  assert.equal(kept, previous, "the same object, so React keeps every line element");
});

test("crossing the hysteresis band does move it", () => {
  const previous: LineWindow = { first: 470, last: 559 };
  const slack = Math.ceil((600 / 20) * HYSTERESIS_VIEWPORTS);
  const viewportTop = (previous.first - slack - 1) * 20;
  const moved = selectLineWindow(geometry({ viewportTop, previous }));
  assert.ok(moved);
  assert.notEqual(moved, previous, "the window moved");
  assert.ok(moved.first < previous.first, "and it moved towards the reader");
});

test("hysteresis cannot pin a window that no longer covers the reader", () => {
  const previous: LineWindow = { first: 0, last: 89 };
  const jumped = selectLineWindow(geometry({ viewportTop: 300_000, previous }));
  assert.ok(jumped);
  assert.notEqual(jumped, previous);
  assert.ok(jumped.first > 14_000, "the window followed the jump");
});

test("the window itself never exceeds the cap", () => {
  const window = selectLineWindow(
    geometry({
      lineCount: 40_000,
      viewportHeight: WINDOW_CAP_LINES * 20,
      viewportTop: 100_000,
      cap: WINDOW_CAP_LINES,
    }),
  );
  assert.ok(window);
  assert.ok(
    window.last - window.first + 1 <= WINDOW_CAP_LINES,
    `window held ${window.last - window.first + 1} lines`,
  );
  assert.ok(window.first <= 5_000 && window.last >= 5_000);
});

test("geometry that cannot be trusted degrades to highlighting everything", () => {
  for (const broken of [
    { lineHeight: 0 },
    { lineHeight: -20 },
    { lineHeight: Number.NaN },
    { viewportHeight: 0 },
    { contentTop: Number.NaN },
    { viewportTop: Number.POSITIVE_INFINITY },
  ]) {
    assert.equal(
      selectLineWindow(geometry(broken)),
      null,
      `geometry ${JSON.stringify(broken)} must not produce a window`,
    );
  }
});

test("a null window highlights every line", () => {
  assert.equal(lineIsWindowed(null, 0), true);
  assert.equal(lineIsWindowed(null, 19_999), true);
});

test("membership is inclusive at both ends", () => {
  const window: LineWindow = { first: 10, last: 20 };
  assert.equal(lineIsWindowed(window, 9), false);
  assert.equal(lineIsWindowed(window, 10), true);
  assert.equal(lineIsWindowed(window, 20), true);
  assert.equal(lineIsWindowed(window, 21), false);
});

test("the window module is plain TypeScript", () => {
  const source = readSrc("components/assistant-ui/code-fence-window.ts");
  assert.ok(
    !/<[A-Za-z]/.test(source.replace(/^\s*[/*].*$/gm, "")),
    "no JSX in the window module",
  );
  assert.ok(
    !/\bfrom\s+["']react["']/.test(source),
    "and no react import: this module has to load under --experimental-strip-types",
  );
});
