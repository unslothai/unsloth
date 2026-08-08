// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The card stores an absolute viewport position, so a position saved on a wide
// monitor is meaningless on a laptop screen. Nothing else can rescue it: the
// card is position:fixed, so an off-screen one creates no scroll to reach it,
// and its own drag handle and collapse button go with it.
//
// Two separate guards, and they are needed together:
//   - the read is clamped, so it can never PAINT off screen, and
//   - the reclamp effect is wired to the panel node, so it keeps up afterwards.
// The second is the one that regressed: the effect captured panelRef.current
// once, while the card was still returning null for an empty list, so it always
// saw null, never built a ResizeObserver, and never re-ran once the node
// existed. Both are asserted here -- the geometry directly, the wiring by
// reading the source, since the node suite has no DOM to mount into.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { clampToViewport } from "../src/features/loaded-models/use-drag-position.ts";

const SOURCE = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/loaded-models/use-drag-position.ts",
      import.meta.url,
    ),
  ),
  "utf8",
);

const CARD = { width: 268, height: 160 };
const LAPTOP = { width: 1280, height: 800 };

// What the initialiser does: clamp before the first paint, with no measurement
// available yet, so zero width and height.
function restore(
  stored: { left: number; top: number },
  viewport: { width: number; height: number },
) {
  return clampToViewport(stored, 0, 0, viewport);
}

test("a position saved on a wider monitor lands back on screen", () => {
  // Dragged to the bottom-right of a 2560x1440 display, reopened on a laptop.
  const restored = restore({ left: 2280, top: 1250 }, LAPTOP);
  assert.ok(restored.left < LAPTOP.width, "must be within the viewport");
  assert.ok(restored.top < LAPTOP.height, "must be within the viewport");
  assert.deepEqual(restored, { left: 1272, top: 792 });
});

test("a position already on screen is left exactly where it was", () => {
  // The common case must not drift by a pixel, or the card would creep each
  // time the app opens.
  const stored = { left: 900, top: 400 };
  assert.deepEqual(restore(stored, LAPTOP), stored);
});

test("a negative stored position is pulled back to the margin", () => {
  // Reachable by dragging on a multi-monitor desktop where the secondary screen
  // sits left of or above the primary.
  assert.deepEqual(restore({ left: -400, top: -90 }, LAPTOP), {
    left: 8,
    top: 8,
  });
});

test("once measured, the whole card is kept on screen, not just its corner", () => {
  // The initialiser clamps with zero size because nothing has been laid out
  // yet; the first ResizeObserver delivery refines it using the real box.
  const corner = restore({ left: 2280, top: 1250 }, LAPTOP);
  const measured = clampToViewport(corner, CARD.width, CARD.height, LAPTOP);
  assert.deepEqual(measured, {
    left: LAPTOP.width - CARD.width - 8,
    top: LAPTOP.height - CARD.height - 8,
  });
});

test("a viewport narrower than the card still leaves it reachable", () => {
  // A phone-width window, or a desktop window dragged very small.
  const tiny = { width: 200, height: 300 };
  const restored = clampToViewport({ left: 900, top: 900 }, CARD.width, 400, tiny);
  assert.deepEqual(restored, { left: 8, top: 8 });
});

test("clamping is idempotent, so the observer cannot feed itself", () => {
  // reclamp() returns the identical object when nothing moved, which is what
  // stops the ResizeObserver -> setPosition -> resubscribe loop from spinning.
  const once = clampToViewport({ left: 5000, top: 5000 }, CARD.width, CARD.height, LAPTOP);
  const twice = clampToViewport(once, CARD.width, CARD.height, LAPTOP);
  assert.deepEqual(once, twice);
});

// ── The wiring the geometry depends on ──────────────────────────────────

test("the stored position is clamped as it is read", () => {
  // Without this the card paints once at the stored coordinates. On a smaller
  // screen that single frame is off screen, and if the observer ever fails to
  // attach it stays there.
  assert.match(
    SOURCE,
    /const stored = readStored[\s\S]{0,500}?clampToViewport\(stored,/,
    "useState initialiser must clamp what it reads",
  );
});

test("the reclamp effect re-runs when the panel node appears", () => {
  // The card renders nothing until the first poll returns a row, so the effect's
  // first run sees no node. A RefObject mutation does not re-render, so the node
  // has to arrive through state for the effect to ever see it.
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
  // WebKitGTK old enough to lack it would otherwise get no clamp at all, and
  // this file's siblings already ponyfill for exactly that vintage.
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
  // Without capture a pointerup over another window is never delivered and the
  // card follows the cursor afterwards.
  assert.match(SOURCE, /setPointerCapture\(event\.pointerId\)/);
  assert.match(
    SOURCE,
    /event\.buttons === 0/,
    "and a move with no button held must end the drag",
  );
});

// The expanded card's grip and the collapsed pill share one drag sentinel, but
// only the pill has a click to consume it. A drag by the grip therefore left
// the flag set, and the pill's next click read someone else's drag and refused
// to expand, costing the user a second click. Asserted by reading the source,
// since the node suite has no DOM to drag in.
const INDICATOR = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/loaded-models/loaded-models-indicator.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
);

test("the expanded grip declares that no click follows its drag", () => {
  const grip = INDICATOR.slice(
    INDICATOR.indexOf('aria-label="Drag to move"'),
    INDICATOR.indexOf('aria-label="Drag to move"') + 400,
  );
  assert.match(
    grip,
    /clickFollows: false/,
    "a handle with no onClick must not leave the sentinel set",
  );
});

test("the collapsed pill still consumes the sentinel itself", () => {
  // It is a button, so its own click reads the flag; marking it clickFollows
  // false would clear the drag before the click could suppress the expand.
  const pill = INDICATOR.slice(
    INDICATOR.indexOf("Show details, or drag to move"),
    INDICATOR.indexOf("Show details, or drag to move") + 400,
  );
  assert.match(pill, /onPointerDown=\{startDrag\}/);
  assert.match(pill, /if \(!justDragged\(\)\) setCollapsed\(false\)/);
});

test("every pointer end path drops a sentinel no click will read", () => {
  const HOOK = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/loaded-models/use-drag-position.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  // `settle` is the single exit: pointerup, pointercancel, and the buttons === 0
  // bail for a release the window never saw all route through it, so clearing
  // there covers each of them and cannot miss one added later.
  const settleAt = HOOK.indexOf("const settle = useCallback(");
  assert.notEqual(settleAt, -1, "expected a settle helper");
  const settle = HOOK.slice(
    settleAt,
    HOOK.indexOf("}, [applyPending]);", settleAt),
  );
  assert.match(settle, /if \(!clickFollowsRef\.current\) movedRef\.current = false;/);
  const onEnd = HOOK.slice(
    HOOK.indexOf("const onEnd = useCallback("),
    HOOK.indexOf("const onEnd = useCallback(") + 260,
  );
  assert.match(onEnd, /settle\(\);/);
  assert.match(HOOK, /if \(event\.buttons === 0\) \{\s*settle\(\);/);
});
