// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Notes are not wire obstacles: moving or resizing one never reroutes a wire.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { isWireObstacle, routingKey } from "../src/features/flows/editor/wire-obstacles.ts";
import { routeWires } from "../src/features/flows/editor/wire-router.ts";

const card = (id: string, x: number, y = 0) => ({
  id,
  type: "flowNode",
  position: { x, y },
  measured: { width: 260, height: 120 },
});
const note = (x: number, y: number, width = 240, height = 140) => ({
  id: "note:n1",
  type: "flowNote",
  position: { x, y },
  measured: { width, height },
});

test("moving or resizing a note leaves the routing key alone; moving a card changes it", () => {
  const base = routingKey([card("a", 0), card("b", 600), note(300, 0)]);
  assert.equal(routingKey([card("a", 0), card("b", 600), note(320, 40)]), base);
  assert.equal(routingKey([card("a", 0), card("b", 600), note(300, 0, 400, 300)]), base);
  assert.notEqual(routingKey([card("a", 10), card("b", 600), note(300, 0)]), base);
  assert.equal(isWireObstacle("flowNote"), false);
  assert.equal(isWireObstacle("flowNode"), true);
});

test("the router never sees a note, so a note on the straight path does not bend the wire", () => {
  const rects = [
    { id: "a", x: 0, y: 0, width: 260, height: 120 },
    { id: "b", x: 700, y: 0, width: 260, height: 120 },
  ];
  const edges = [{ id: "e", source: "a", target: "b", sourceSide: "right", targetSide: "left", sourceOffset: 60, targetOffset: 60 }] as Parameters<typeof routeWires>[1];
  const straight = routeWires(rects, edges).routes.get("e");
  assert.ok(straight);
  assert.equal(new Set(straight.points.map((p) => Math.round(p.y))).size, 1, "a straight run between level ports");
  const src = readFileSync(new URL("../src/features/flows/editor/wire-routing.tsx", import.meta.url), "utf8");
  assert.match(src, /node\.hidden \|\| !isWireObstacle\(node\.type\)/);
  assert.match(src, /useStore\(\(s\) => routingKey\(\[\.\.\.s\.nodeLookup\.values\(\)\]\)\)/);
  assert.match(src, /\[nodesKey, edges, scope, store(, moving)?(, global)?\]/);
});

test("turning a reroute around reroutes its wires though its box is unchanged", () => {
  const rr = (dir: string) => ({
    id: "rr1",
    type: "flowReroute",
    position: { x: 100, y: 100 },
    measured: { width: 16, height: 28 },
    data: { rerouteId: "rr1", dir },
  });
  assert.notEqual(routingKey([card("a", 0), rr("down")]), routingKey([card("a", 0), rr("up")]));
});

test("an R typed in a field while the reroute tool is armed stays in the field", () => {
  const src = readFileSync(
    new URL("../src/features/flows/editor/canvas-controls.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /!isTyping\(e\.target\) &&\s*armed === REROUTE_TYPE &&/);
});
