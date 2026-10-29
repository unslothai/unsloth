// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A wire from a card below and left of its target never runs through a card, before or after a resize.

import assert from "node:assert/strict";
import test from "node:test";

import { routeWires } from "../src/features/flows/editor/wire-router.ts";

type R = { id: string; x: number; y: number; width: number; height: number };
const crosses = (a: { x: number; y: number }, b: { x: number; y: number }, r: R) => {
  // Strictly inside the card, so a leg along the edge or the port stub does not count.
  const x1 = Math.min(a.x, b.x), x2 = Math.max(a.x, b.x), y1 = Math.min(a.y, b.y), y2 = Math.max(a.y, b.y);
  return x2 > r.x + 1 && x1 < r.x + r.width - 1 && y2 > r.y + 1 && y1 < r.y + r.height - 1;
};
const clear = (points: { x: number; y: number }[], rects: R[]) =>
  points.every((p, i) => i === 0 || rects.every((r) => !crosses(points[i - 1], p, r)));
const wire = { id: "e", source: "t", target: "g", sourceSide: "right", targetSide: "left", sourceOffset: 66, targetOffset: 66, net: "t:text", sink: "g:prompt" } as Parameters<typeof routeWires>[1][number];

for (const gap of [90, 60, 40]) {
  test(`source below and left, target above and right, ${gap}px gap: routed round both cards`, () => {
    const g: R = { id: "g", x: 260, y: 0, width: 320, height: 470 };
    const t: R = { id: "t", x: 0, y: 470 + gap, width: 440, height: 400 };
    const route = routeWires([g, t], [wire]).routes.get("e");
    assert.ok(route);
    assert.equal(route.blocked, false);
    assert.ok(clear(route.points, [g, t]), JSON.stringify(route.points));
  });
}

test("after the source card grows over its old route, the next pass routes round it", () => {
  const g: R = { id: "g", x: 260, y: 0, width: 320, height: 470 };
  const small: R = { id: "t", x: 0, y: 560, width: 260, height: 120 };
  const first = routeWires([g, small], [wire]);
  const grown: R = { ...small, width: 440, height: 400 };
  const route = routeWires([g, grown], [wire], {}, first.cache).routes.get("e");
  assert.ok(route);
  assert.ok(clear(route.points, [g, grown]), JSON.stringify(route.points));
});
