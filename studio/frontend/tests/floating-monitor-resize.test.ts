// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  FLOATING_PANEL_POSITION_KEYS,
  readPanelWidth,
  resizedWidth,
} from "../src/hooks/floating-panel-position.ts";
import { readSrc } from "./helpers/kit.ts";

const BOUNDS = { min: 232, max: 520 };
const PANEL = { width: 256 };

test("a side edge follows the pointer, mirrored on the left", () => {
  const right = { ...PANEL, edges: { x: 1, y: 0 } } as const;
  const left = { ...PANEL, edges: { x: -1, y: 0 } } as const;
  assert.equal(resizedWidth(right, 40, BOUNDS), 296);
  assert.equal(resizedWidth(left, -40, BOUNDS), 296);
  assert.equal(resizedWidth(left, 40, BOUNDS), 232);
});

test("a corner only changes the width: the height follows the content", () => {
  const corner = { ...PANEL, edges: { x: 1, y: 1 } } as const;
  assert.equal(resizedWidth(corner, 40, BOUNDS), 296);
  const top = { ...PANEL, edges: { x: 0, y: -1 } } as const;
  assert.equal(resizedWidth(top, 40, BOUNDS), 256);
});

test("width stays inside min and max, and the max yields to a smaller min", () => {
  const right = { ...PANEL, edges: { x: 1, y: 0 } } as const;
  assert.equal(resizedWidth(right, 1000, BOUNDS), 520);
  assert.equal(resizedWidth(right, -1000, BOUNDS), 232);
  // No room at all (a corner against the viewport edge) keeps the floor, never below it.
  assert.equal(resizedWidth(right, 50, { min: 232, max: 100 }), 232);
});

test("a stored width is clamped, and junk is ignored", () => {
  const store = new Map<string, string>();
  const previous = globalThis.localStorage;
  globalThis.localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
  } as Storage;
  try {
    const options = { minWidth: 232, maxWidth: 520, defaultWidth: () => 256, storageKey: "w" };
    assert.equal(readPanelWidth(options), null);
    store.set("w", "300");
    assert.equal(readPanelWidth(options), 300);
    store.set("w", "9000");
    assert.equal(readPanelWidth(options), 520);
    store.set("w", "abc");
    assert.equal(readPanelWidth(options), null);
    assert.equal(readPanelWidth(undefined), null);
  } finally {
    globalThis.localStorage = previous;
  }
});

test("the stored width is cleared with the stored positions", () => {
  assert.equal(FLOATING_PANEL_POSITION_KEYS.liveMonitorWidth, "unsloth_live_monitor_width");
  const monitor = readSrc("components/floating-monitor.tsx");
  assert.match(monitor, /storageKey: FLOATING_PANEL_POSITION_KEYS\.liveMonitorWidth/);
  assert.match(monitor, /data-edge=\{edge\}/);
  assert.doesNotMatch(monitor, /panel-scale/);
});

test("drag, nudge and resize never run at once, and a squeezed monitor grows back to its default width", () => {
  const hook = readSrc("hooks/use-floating-panel-layout.ts");
  for (const name of ["startDrag", "nudge", "startResize"]) {
    const body = hook.slice(hook.indexOf(`function ${name}(`)).slice(0, 600);
    assert.match(body, /dragSessionRef\.current \|\|\s*resizeSessionRef\.current/, name);
  }
  assert.match(hook, /widthRef\.current \?\? resizeRef\.current\.defaultWidth\(\)/);
});

test("a drag never clamps the chosen width of a squeezed resizable panel", () => {
  const hook = readSrc("hooks/use-floating-panel-layout.ts");
  const drag = hook.slice(hook.indexOf("function startDrag("), hook.indexOf("dragSessionRef.current = {"));
  assert.match(drag, /!resizeRef\.current &&\s*Number\.isFinite\(inlineWidth\)/);
});
