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
const PANEL = { width: 256, height: 128 };

test("a side edge follows the pointer, mirrored on the left", () => {
  const right = { ...PANEL, edges: { x: 1, y: 0 } } as const;
  const left = { ...PANEL, edges: { x: -1, y: 0 } } as const;
  assert.equal(resizedWidth(right, 40, 0, BOUNDS), 296);
  assert.equal(resizedWidth(left, -40, 0, BOUNDS), 296);
  assert.equal(resizedWidth(left, 40, 0, BOUNDS), 232);
});

test("a top or bottom edge scales width by the panel's aspect, so the card grows taller", () => {
  const top = { ...PANEL, edges: { x: 0, y: -1 } } as const;
  // 20px up on a 2:1 card is 40px wider.
  assert.equal(resizedWidth(top, 0, -20, BOUNDS), 296);
  const bottom = { ...PANEL, edges: { x: 0, y: 1 } } as const;
  assert.equal(resizedWidth(bottom, 0, 20, BOUNDS), 296);
});

test("a corner takes whichever axis moved further", () => {
  const corner = { ...PANEL, edges: { x: 1, y: 1 } } as const;
  assert.equal(resizedWidth(corner, 40, 0, BOUNDS), 296);
  assert.equal(resizedWidth(corner, 10, 30, BOUNDS), 316);
});

test("width stays inside min and max, and the max yields to a smaller min", () => {
  const right = { ...PANEL, edges: { x: 1, y: 0 } } as const;
  assert.equal(resizedWidth(right, 1000, 0, BOUNDS), 520);
  assert.equal(resizedWidth(right, -1000, 0, BOUNDS), 232);
  // No room at all (a corner against the viewport edge) keeps the floor, never below it.
  assert.equal(resizedWidth(right, 50, 0, { min: 232, max: 100 }), 232);
});

test("a stored width is clamped, and junk is ignored", () => {
  const store = new Map<string, string>();
  const previous = globalThis.localStorage;
  globalThis.localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
  } as Storage;
  try {
    const options = { baseWidth: 256, ...{ minWidth: 232, maxWidth: 520 }, storageKey: "w" };
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
  assert.match(monitor, /\[zoom:var\(--panel-scale,1\)\]/);
});
