// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Placing cards from the Nodes panel: arming, where a card lands, grid snap, and which spots take a card.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  GRAB,
  GRID,
  PLACE_GROUP,
  PLACE_NOTE,
  PLACE_SHAPES,
  placeableTarget,
  placedRect,
  placementPosition,
  snapPoint,
  stopPlacement,
  togglePlacement,
  usePlacement,
} = await import("../src/features/flows/editor/placement.ts");
const { REROUTE_SIZE } = await import("../src/features/flows/editor/reroutes.ts");
const { nearestOnPath } = await import("../src/features/flows/editor/wire-snap.ts");

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

test("stopping placement disarms, and stopping twice is harmless", () => {
  togglePlacement("text.input");
  stopPlacement();
  assert.equal(armed(), null);
  stopPlacement();
  assert.equal(armed(), null);
});

test("a placed node lands with its header under the pointer, on the grid when snap is on", () => {
  assert.deepEqual(placementPosition({ x: 500, y: 300 }, false), [473, 273]);
  assert.deepEqual(placementPosition({ x: 507.4, y: 301 }, true), [480, 280]);
  const [x, y] = placementPosition({ x: -45, y: 13 }, true);
  assert.equal(Math.abs(x % 20), 0);
  assert.equal(Math.abs(y % 20), 0);
});

test("a placed card's header icon lands on the pointer, in flow units, at any spot", () => {
  for (const flow of [{ x: 0, y: 0 }, { x: 123.4, y: -56.7 }, { x: -980.2, y: 441.9 }]) {
    const [x, y] = placementPosition(flow, false);
    assert.ok(Math.hypot(x + GRAB[0] - flow.x, y + GRAB[1] - flow.y) <= 1, "unsnapped: within rounding");
    const [sx, sy] = placementPosition(flow, true);
    assert.ok(Math.abs(sx + GRAB[0] - flow.x) <= 10 && Math.abs(sy + GRAB[1] - flow.y) <= 10, "snapped: within half a grid step");
  }
});

test("a reroute is centred on the pointer, groups and notes hang from it", () => {
  assert.deepEqual(PLACE_SHAPES.reroute.size, [REROUTE_SIZE.width, REROUTE_SIZE.height]);
  const rect = placedRect("reroute", { x: 203, y: 97 }, true);
  assert.ok(rect);
  const [x, y, w, h] = rect;
  assert.deepEqual([x + w / 2, y + h / 2], [200, 100]);
  assert.deepEqual(placedRect(PLACE_NOTE, { x: 300, y: 300 }, false), [280, 280, 240, 140]);
  assert.deepEqual(placedRect(PLACE_GROUP, { x: 300, y: 300 }, false)?.slice(0, 2), [280, 280]);
  assert.equal(placedRect("text.input", { x: 0, y: 0 }, true), null);
});

test("a reroute dropped near a wire lands on the wire's nearest point", () => {
  // An L-shaped wire: (0,0) -> (100,0) -> (100,100).
  const at = (d: number) => (d <= 100 ? { x: d, y: 0 } : { x: 100, y: d - 100 });
  const p = nearestOnPath(200, at, { x: 40, y: 7 });
  assert.ok(Math.abs(p.x - 40) < 0.5 && p.y === 0, JSON.stringify(p));
  const q = nearestOnPath(200, at, { x: 108, y: 63 });
  assert.ok(q.x === 100 && Math.abs(q.y - 63) < 0.5, JSON.stringify(q));
});

// Snap to grid lands a card's top-left on the canvas dots.

test("dots sit on multiples of the grid at any pan and zoom", () => {
  // @xyflow/react Background, dots: tile at pan % (gap*zoom), translated by -(offset*zoom),
  // circle at (radius, radius) with radius = size*zoom/2.
  const size = 1.2;
  const offset = size / 2;
  for (const [pan, zoom] of [[0, 1], [37, 1], [-113.5, 0.5], [512, 1.75]]) {
    const tile = GRID * zoom;
    for (let k = -2; k <= 2; k++) {
      const screen = (pan % tile) + k * tile - offset * zoom + (size * zoom) / 2;
      const flow = (screen - pan) / zoom;
      const nearest = Math.round(flow / GRID) * GRID;
      assert.ok(Math.abs(flow - nearest) < 1e-9, `pan ${pan} zoom ${zoom}: dot at ${flow}`);
    }
  }
});

test("snapPoint rounds to the 20px grid when snap is on, to whole pixels when off", () => {
  assert.equal(GRID, 20);
  assert.deepEqual(snapPoint([29, -31], true), [20, -40]);
  assert.deepEqual(snapPoint([29.4, -31.6], false), [29, -32]);
});

// An armed Nodes panel item previews and places over a group or a note, not just bare canvas.

const el = (classes: string[], ancestors: string[] = []) => ({
  classList: { contains: (c: string) => classes.includes(c) },
  closest: (sel: string) => {
    const all = [...classes, ...ancestors];
    return sel.split(",").some((part) => {
      const s = part.trim();
      if (s.startsWith(".")) return all.includes(s.slice(1));
      return all.includes(`<${s.replace(/\[.*\]/, "")}>`);
    })
      ? {}
      : null;
  },
});

const targets: [string, ReturnType<typeof el> | null, boolean][] = [
  ["bare canvas", el(["react-flow__pane"]), true],
  ["a group's body", el(["group/frame"], ["react-flow__node-flowGroup"]), true],
  ["a note's body", el(["text-xs"], ["react-flow__node-flowNote"]), true],
  ["a card", el(["flex"], ["react-flow__node-flowNode"]), false],
  ["a button on a group", el(["size-6"], ["<button>", "react-flow__node-flowGroup"]), false],
  ["no target", null, false],
];
for (const [name, target, placeable] of targets) {
  test(`placeableTarget: ${name} ${placeable ? "takes" : "does not take"} the card`, () => {
    assert.equal(placeableTarget(target as never), placeable);
  });
}
