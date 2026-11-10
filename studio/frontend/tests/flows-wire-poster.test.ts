// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A wire from a card below and left of its target never runs through a card, before or after a resize.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { routeWires } = await import("../src/features/flows/editor/wire-router.ts");
type RouteRequest = import("../src/features/flows/editor/wire-router.worker.ts").RouteRequest;
type RouteResponse = import("../src/features/flows/editor/wire-router.worker.ts").RouteResponse;

// The routing worker's own message handler, run in-process: `self` is the worker scope.
const replies: RouteResponse[] = [];
const workerScope: { onmessage: ((event: { data: RouteRequest }) => void) | null; postMessage: (m: RouteResponse) => void } = {
  onmessage: null,
  postMessage: (m) => void replies.push(m),
};
(globalThis as Record<string, unknown>).self = workerScope;
await import("../src/features/flows/editor/wire-router.worker.ts");
type Wire = Parameters<typeof routeWires>[1][number];
let nextRequest = 0;
const ask = (rects: R[], edges: Wire[], partial = false, scope = "flow") => {
  workerScope.onmessage?.({
    data: { id: ++nextRequest, scope, rects, zones: [], edges, hopRadius: 4, partial },
  });
  const reply = replies.at(-1);
  assert.equal(reply?.id, nextRequest);
  assert.equal(reply?.error, undefined);
  return reply?.routes;
};

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

test("the same document routes the same way every time: full passes start from scratch", () => {
  const g: R = { id: "g", x: 260, y: 0, width: 320, height: 470 };
  const t: R = { id: "t", x: 0, y: 560, width: 440, height: 400 };
  const a = routeWires([g, t], [wire]).routes.get("e");
  const b = routeWires([g, t], [wire]).routes.get("e");
  assert.deepEqual(a, b);
});

test("the routing worker: a full pass after other passes routes as from scratch, a partial one reuses the last", () => {
  // Two wires sharing a corridor; moving c changes how a cached pass and a fresh one lay them out.
  const card = (id: string, x: number, y: number): R => ({ id, x, y, width: 220, height: 120 });
  const pair = (source: string, target: string): Wire => ({
    id: `${source}->${target}`, source, target, sourceSide: "right", targetSide: "left", net: source,
  });
  const edges = [pair("a", "d"), pair("b", "c")];
  const before = [card("a", 0, 0), card("b", 0, 200), card("c", 600, 100), card("d", 600, 400)];
  const after = before.map((r) => (r.id === "c" ? { ...r, y: r.y + 100 } : r));
  const fresh = [...routeWires(after, edges, { hopRadius: 4 }).routes.values()];
  const reused = [
    ...routeWires(after, edges, { hopRadius: 4 }, routeWires(before, edges, { hopRadius: 4 }).cache).routes.values(),
  ];
  assert.notDeepEqual(reused, fresh, "the scene tells a cached pass from a fresh one");
  ask(before, edges);
  assert.deepEqual(ask(after, edges), fresh, "full pass: same as a fresh document");
  ask(before, edges);
  assert.deepEqual(ask(after, edges, true), reused, "partial pass: reuses the last result");
  ask(before, edges, false, "other flow");
  assert.deepEqual(ask(after, edges, true), fresh, "another flow's cache is dropped");
});
