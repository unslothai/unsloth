// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #14330: a clamp (e.g. the 0x0 Windows minimize) became the panel's new home.

import assert from "node:assert/strict";
import test from "node:test";

import { clampToViewport } from "../src/features/loaded-models/use-drag-position.ts";
import {
  FLOATING_PANEL_POSITION_KEYS,
  isDegenerateBox,
  placeAxis,
  readPanelIntent,
  storePanelIntent,
} from "../src/hooks/floating-panel-position.ts";

import { readSrc } from "./helpers/kit.ts";

function withStorage(run: (store: Map<string, string>) => void): void {
  const store = new Map<string, string>();
  const previous = (globalThis as { localStorage?: unknown }).localStorage;
  (globalThis as { localStorage?: unknown }).localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => store.set(key, value),
    removeItem: (key: string) => store.delete(key),
  };
  try {
    run(store);
  } finally {
    (globalThis as { localStorage?: unknown }).localStorage = previous;
  }
}

test("a minimized window is a degenerate box and a real one is not", () => {
  assert.equal(isDegenerateBox({ width: 0, height: 0 }), true);
  assert.equal(isDegenerateBox({ width: 1200, height: 0 }), true);
  assert.equal(isDegenerateBox({ width: 0, height: 800 }), true);
  assert.equal(isDegenerateBox({ width: 1, height: 1 }), false);
  assert.equal(isDegenerateBox({ width: 1440, height: 900 }), false);
});

test("an undragged axis stays anchored to the far edge", () => {
  assert.equal(placeAxis(null, 900), 900);
  assert.equal(placeAxis(null, 0), 0);
});

test("a shrink clamps the painted spot and growing back restores it", () => {
  const intent = 1100;
  const sizes = [1200, 600, 1200];
  assert.deepEqual(
    sizes.map((max) => placeAxis(intent, max)),
    [1100, 600, 1100],
  );
});

test("the card returns to its spot after a minimize and restore", () => {
  const card = { width: 268, height: 160 };
  const chosen = { left: 900, top: 500 };
  const big = { width: 1440, height: 900 };
  const minimized = clampToViewport(chosen, card.width, card.height, { width: 0, height: 0 });
  assert.deepEqual(minimized, { left: 8, top: 8 });
  assert.deepEqual(clampToViewport(minimized, card.width, card.height, big), { left: 8, top: 8 });
  assert.deepEqual(clampToViewport(chosen, card.width, card.height, big), chosen);
});

test("a dropped spot round-trips through storage, anchors included", () => {
  withStorage(() => {
    const key = FLOATING_PANEL_POSITION_KEYS.liveMonitor;
    assert.equal(readPanelIntent(key), null);
    storePanelIntent(key, { left: 420, top: null });
    assert.deepEqual(readPanelIntent(key), { left: 420, top: null });
    storePanelIntent(key, { left: 12.5, top: 300 });
    assert.deepEqual(readPanelIntent(key), { left: 12.5, top: 300 });
  });
});

test("a corrupt stored spot falls back to the default placement", () => {
  withStorage((store) => {
    const key = FLOATING_PANEL_POSITION_KEYS.apiMonitor;
    for (const raw of ["{", "null", "[]", '{"left":"5","top":1}', '{"left":1}', '{"left":1e999,"top":1}']) {
      store.set(key, raw);
      assert.equal(readPanelIntent(key), null, raw);
    }
    assert.equal(readPanelIntent(undefined), null);
  });
});

test("both monitors persist under their own key and a reset clears them", () => {
  assert.match(
    readSrc("components/floating-monitor.tsx"),
    /FLOATING_PANEL_POSITION_KEYS\.liveMonitor,\s*cornerCard,\s*\);/,
  );
  assert.match(
    readSrc("features/api-monitor/api-monitor-overlay.tsx"),
    /FLOATING_PANEL_POSITION_KEYS\.apiMonitor,\s*\);/,
  );
  assert.match(
    readSrc("features/settings/tabs/general-tab.tsx"),
    /\.\.\.Object\.values\(FLOATING_PANEL_POSITION_KEYS\),/,
  );
});

test("layout skips a 0x0 frame and only a drag or a nudge writes the stored spot", () => {
  const hook = readSrc("hooks/use-floating-panel-layout.ts");
  assert.match(hook, /if \(isDegenerateBox\(constraintsBox\)\) \{\s*return;/);
  assert.equal(hook.split("storePanelIntent(").length - 1, 2);
  const nudge = hook.slice(hook.indexOf("function nudge("));
  assert.match(nudge, /storePanelIntent\(storageKey, intentRef\.current\)/);
  const finish = hook.slice(hook.indexOf("function finishDrag"));
  assert.match(finish, /storePanelIntent\(storageKey, intentRef\.current\)/);
  const card = readSrc("features/loaded-models/use-drag-position.ts");
  assert.match(card, /if \(isDegenerateBox\(view\)\) return;/);
});
