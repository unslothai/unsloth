// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The card is position:fixed, so a stored off-screen position is unreachable; the read is
// clamped and the reclamp effect must attach once the panel node exists.

import assert from "node:assert/strict";
import test from "node:test";

import { clampToViewport } from "../src/features/loaded-models/use-drag-position.ts";

import { readSrc } from "./helpers/kit.ts";

const USE_DRAG_POSITION = readSrc("features/loaded-models/use-drag-position.ts");

const SOURCE = readSrc("features/loaded-models/use-drag-position.ts");

const CARD = { width: 268, height: 160 };
const LAPTOP = { width: 1280, height: 800 };

function restore(
  stored: { left: number; top: number },
  viewport: { width: number; height: number },
) {
  return clampToViewport(stored, 0, 0, viewport);
}

test("a position saved on a wider monitor lands back on screen", () => {
  const restored = restore({ left: 2280, top: 1250 }, LAPTOP);
  assert.ok(restored.left < LAPTOP.width, "must be within the viewport");
  assert.ok(restored.top < LAPTOP.height, "must be within the viewport");
  assert.deepEqual(restored, { left: 1272, top: 792 });
});

test("a position already on screen is left exactly where it was", () => {
  // In-bounds positions must not drift, or the card creeps on every open.
  const stored = { left: 900, top: 400 };
  assert.deepEqual(restore(stored, LAPTOP), stored);
});

test("a negative stored position is pulled back to the margin", () => {
  assert.deepEqual(restore({ left: -400, top: -90 }, LAPTOP), {
    left: 8,
    top: 8,
  });
});

test("once measured, the whole card is kept on screen, not just its corner", () => {
  const corner = restore({ left: 2280, top: 1250 }, LAPTOP);
  const measured = clampToViewport(corner, CARD.width, CARD.height, LAPTOP);
  assert.deepEqual(measured, {
    left: LAPTOP.width - CARD.width - 8,
    top: LAPTOP.height - CARD.height - 8,
  });
});

test("a viewport narrower than the card still leaves it reachable", () => {
  const tiny = { width: 200, height: 300 };
  const restored = clampToViewport({ left: 900, top: 900 }, CARD.width, 400, tiny);
  assert.deepEqual(restored, { left: 8, top: 8 });
});

test("clamping is idempotent, so the observer cannot feed itself", () => {
  // Returning the same object stops a ResizeObserver/setPosition loop.
  const once = clampToViewport({ left: 5000, top: 5000 }, CARD.width, CARD.height, LAPTOP);
  const twice = clampToViewport(once, CARD.width, CARD.height, LAPTOP);
  assert.deepEqual(once, twice);
});

test("the stored position is clamped as it is read", () => {
  assert.match(
    SOURCE,
    /const stored = readStored[\s\S]{0,500}?clampToViewport\(stored,/,
    "useState initialiser must clamp what it reads",
  );
});

test("the reclamp effect re-runs when the panel node appears", () => {
  // The node arrives via state because a RefObject mutation does not re-render.
  const guard = SOURCE.indexOf("!panelEl) return;");
  assert.notEqual(
    guard,
    -1,
    "the reclamp effect must guard on the panel node, not read a ref",
  );
  const effect = SOURCE.slice(guard);
  const deps = effect.slice(0, effect.indexOf("]") + 1);
  assert.match(
    deps,
    /\bpanelEl\b/,
    "panelEl must be a dependency or the effect never re-subscribes",
  );
  assert.ok(
    !/const panel = panelRef\.current;\s*\n\s*const measure/.test(SOURCE),
    "the effect must not snapshot the ref, which is null on its first run",
  );
  assert.match(
    SOURCE,
    /setPanelEl\(node\)/,
    "the ref has to be a callback that sets state",
  );
});

test("a missing ResizeObserver still leaves the card clampable", () => {
  // Old WebKitGTK lacks ResizeObserver.
  assert.match(
    SOURCE,
    /typeof ResizeObserver === "undefined"/,
    "construction must be guarded",
  );
  assert.match(
    SOURCE,
    /window\.addEventListener\("resize", measure\)/,
    "the resize path is the fallback and must not be conditional on it",
  );
});

test("the drag captures the pointer", () => {
  // Without capture, a pointerup over another window is never delivered.
  assert.match(SOURCE, /setPointerCapture\(event\.pointerId\)/);
  assert.match(
    SOURCE,
    /event\.buttons === 0/,
    "and a move with no button held must end the drag",
  );
});

// Grip and pill share one drag sentinel; startDrag zeroes it on every pointerdown.
const INDICATOR = readSrc("features/loaded-models/loaded-models-indicator.tsx");

test("every drag handle goes through startDrag, which resets the sentinel", () => {
  const startDrag = USE_DRAG_POSITION.slice(
    USE_DRAG_POSITION.indexOf("const startDrag = useCallback("),
    USE_DRAG_POSITION.indexOf("// One paint per frame"),
  );
  assert.match(startDrag, /movedRef\.current = false;/);
  const handles = INDICATOR.match(/onPointerDown=\{startDrag\}/g);
  assert.equal(handles?.length, 2);
});

test("only the pill consumes the sentinel, since only it has a click", () => {
  const pill = INDICATOR.slice(
    INDICATOR.indexOf("Show details, or drag to move"),
    INDICATOR.indexOf("Show details, or drag to move") + 400,
  );
  assert.match(pill, /onPointerDown=\{startDrag\}/);
  assert.match(pill, /if \(!justDragged\(\)\) setCollapsed\(false\)/);
});

// Only a landed drag persists; reclamps are display-time and must not overwrite the save.
test("only a drag persists a position, never a reclamp", () => {
  const settleAt = USE_DRAG_POSITION.indexOf("const settle = useCallback(");
  const settle = USE_DRAG_POSITION.slice(
    settleAt,
    USE_DRAG_POSITION.indexOf("}, [applyPending, storageKey]);", settleAt),
  );
  assert.match(settle, /store\(storageKey, landed\)/);
  assert.doesNotMatch(USE_DRAG_POSITION, /useEffect\(\(\) => \{\s*if \(pressing\) return;\s*store\(/);
  assert.equal(
    USE_DRAG_POSITION.split("store(storageKey").length - 1,
    1,
    "one write, in settle",
  );
});
