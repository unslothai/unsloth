// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  PLACE_GROUP,
  PLACE_NOTE,
  PLACE_SHAPES,
  placedRect,
  placementPosition,
  stopPlacement,
  togglePlacement,
  usePlacement,
} from "../src/features/flows/editor/placement.ts";

const armed = () => usePlacement.getState().armed;

test("clicking an item arms it, clicking it again disarms, another item switches", () => {
  stopPlacement();
  togglePlacement("text.input");
  assert.equal(armed(), "text.input");
  togglePlacement("llm.generate");
  assert.equal(armed(), "llm.generate");
  togglePlacement("llm.generate");
  assert.equal(armed(), null);
});

test("Esc, right-click and leaving the editor all stop placement", () => {
  togglePlacement("text.input");
  stopPlacement();
  assert.equal(armed(), null);
  stopPlacement();
  assert.equal(armed(), null);
  const src = readFileSync(
    new URL("../src/features/flows/editor/canvas-controls.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /useEffect\(\(\) => stopPlacement, \[\]\);/);
  assert.match(src, /e\.key === "Escape"\) stopPlacement\(\)/);
  assert.match(src, /const menu = [\s\S]*?stopPlacement\(\);/);
});

test("a placed node lands with its header under the pointer, on the grid when snap is on", () => {
  assert.deepEqual(placementPosition({ x: 500, y: 300 }, false), [473, 273]);
  assert.deepEqual(placementPosition({ x: 507.4, y: 301 }, true), [480, 280]);
  const [x, y] = placementPosition({ x: -45, y: 13 }, true);
  assert.equal(Math.abs(x % 20), 0);
  assert.equal(Math.abs(y % 20), 0);
});

test("only placeable spots place (canvas, group or note body); clicks on cards do not", () => {
  const src = readFileSync(
    new URL("../src/features/flows/editor/canvas-controls.tsx", import.meta.url),
    "utf8",
  );
  const click = src.slice(src.indexOf("const click ="), src.indexOf("const dblclick"));
  assert.match(click, /!onPane\(e\)\) return;/);
  assert.match(src, /const onPane = \(e: Event\) => placeableTarget\(e\.target as Element\);/);
});

test("a placed card's header icon lands on the pointer, in flow units, at any spot", async () => {
  const { GRAB } = await import("../src/features/flows/editor/placement.ts");
  for (const flow of [{ x: 0, y: 0 }, { x: 123.4, y: -56.7 }, { x: -980.2, y: 441.9 }]) {
    const [x, y] = placementPosition(flow, false);
    assert.ok(Math.hypot(x + GRAB[0] - flow.x, y + GRAB[1] - flow.y) <= 1, "unsnapped: within rounding");
    const [sx, sy] = placementPosition(flow, true);
    assert.ok(Math.abs(sx + GRAB[0] - flow.x) <= 10 && Math.abs(sy + GRAB[1] - flow.y) <= 10, "snapped: within half a grid step");
  }
});

test("the ghost uses the same grab point and canvas zoom, portaled clear of transformed ancestors", () => {
  const src = readFileSync(new URL("../src/features/flows/editor/placement-ghost.tsx", import.meta.url), "utf8");
  assert.match(src, /left: x - GRAB\[0\] \* zoom,\s*top: y - GRAB\[1\] \* zoom,\s*transform: `scale\(\$\{zoom\}\)`/);
  assert.match(src, /createPortal\(/);
  for (const file of ["canvas-controls.tsx", "palette.tsx"])
    assert.match(readFileSync(new URL(`../src/features/flows/editor/${file}`, import.meta.url), "utf8"), /<PlacementGhost /);
});

test("a reroute is centred on the pointer, groups and notes hang from it", () => {
  const reroutes = readFileSync(new URL("../src/features/flows/editor/reroutes.ts", import.meta.url), "utf8");
  const [, rw, rh] = reroutes.match(/REROUTE_SIZE = \{ width: (\d+), height: (\d+) \}/) ?? [];
  assert.deepEqual(PLACE_SHAPES.reroute.size, [Number(rw), Number(rh)]);
  assert.match(reroutes, /REROUTE_TYPE = "reroute"/);
  const rect = placedRect("reroute", { x: 203, y: 97 }, true);
  assert.ok(rect);
  const [x, y, w, h] = rect;
  assert.deepEqual([x + w / 2, y + h / 2], [200, 100]);
  assert.deepEqual(placedRect(PLACE_NOTE, { x: 300, y: 300 }, false), [280, 280, 240, 140]);
  assert.deepEqual(placedRect(PLACE_GROUP, { x: 300, y: 300 }, false)?.slice(0, 2), [280, 280]);
  assert.equal(placedRect("text.input", { x: 0, y: 0 }, true), null);
});

test("armed toolbar shapes place through placedRect and show their own ghost", () => {
  const src = readFileSync(new URL("../src/features/flows/editor/canvas-controls.tsx", import.meta.url), "utf8");
  assert.match(src, /placeArmed\(\s*armed,\s*screenToFlowPosition/);
  assert.match(src, /<ShapeGhost type=\{armed\}/);
});

test("a reroute dropped near a wire lands on the wire's nearest point", async () => {
  const { nearestOnPath } = await import("../src/features/flows/editor/wire-snap.ts");
  // An L-shaped wire: (0,0) -> (100,0) -> (100,100).
  const at = (d: number) => (d <= 100 ? { x: d, y: 0 } : { x: 100, y: d - 100 });
  const p = nearestOnPath(200, at, { x: 40, y: 7 });
  assert.ok(Math.abs(p.x - 40) < 0.5 && p.y === 0, JSON.stringify(p));
  const q = nearestOnPath(200, at, { x: 108, y: 63 });
  assert.ok(q.x === 100 && Math.abs(q.y - 63) < 0.5, JSON.stringify(q));
});

test("an armed reroute threads into the wire under the pointer, and the wire's × hides", () => {
  const read = (f: string) => readFileSync(new URL(`../src/features/flows/editor/${f}`, import.meta.url), "utf8");
  const controls = read("canvas-controls.tsx");
  assert.match(controls, /addReroute\(\s*\[[\s\S]*?hit\.wire,\s*\)/);
  assert.match(controls, /setPointer\(flowToScreenPosition\(hit\.at\)\)/);
  assert.match(read("wire-delete.tsx"), /if \(!show \|\| rerouting\) return null;/);
});
