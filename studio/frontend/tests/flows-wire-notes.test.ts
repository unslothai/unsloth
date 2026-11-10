// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Notes are not wire obstacles: moving or resizing one never reroutes a wire.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isWireObstacle, routingVersion } = await import("../src/features/flows/editor/wire-obstacles.ts");
const { routeWires } = await import("../src/features/flows/editor/wire-router.ts");
const { buildRequest } = await import("../src/features/flows/editor/wire-schedule.ts");
const { turnsArmedReroute } = await import("../src/features/flows/editor/history-keys.ts");

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

test("moving or resizing a note leaves the routing version alone; moving a card changes it", () => {
  const version = routingVersion();
  const base = version([card("a", 0), card("b", 600), note(300, 0)]);
  assert.equal(version([card("a", 0), card("b", 600), note(320, 40)]), base);
  assert.equal(version([card("a", 0), card("b", 600), note(300, 0, 400, 300)]), base);
  assert.notEqual(version([card("a", 10), card("b", 600), note(300, 0)]), base);
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
  // The request the canvas sends the router leaves notes (and hidden cards) out.
  const placed = (n: ReturnType<typeof card> | ReturnType<typeof note>, hidden = false) => ({
    ...n,
    hidden,
    internals: { positionAbsolute: n.position },
  });
  const lookup = new Map(
    [placed(card("a", 0)), placed(card("b", 700)), placed(note(300, 0)), placed(card("gone", 300), true)].map(
      (n) => [n.id, n] as const,
    ),
  );
  const request = buildRequest(lookup as never, []);
  assert.deepEqual(request.rects.map((r) => r.id).sort(), ["a", "b"]);
});

test("turning a reroute around reroutes its wires though its box is unchanged", () => {
  const rr = (dir: string) => ({
    id: "rr1",
    type: "flowReroute",
    position: { x: 100, y: 100 },
    measured: { width: 16, height: 28 },
    data: { rerouteId: "rr1", dir },
  });
  const version = routingVersion();
  assert.notEqual(version([card("a", 0), rr("down")]), version([card("a", 0), rr("up")]));
});

const field = { tagName: "INPUT" } as unknown as EventTarget;
const canvas = { tagName: "DIV" } as unknown as EventTarget;
const key = (k: string, target: EventTarget, mods: Partial<Record<"metaKey" | "ctrlKey" | "altKey", boolean>> = {}) => ({
  key: k,
  target,
  metaKey: false,
  ctrlKey: false,
  altKey: false,
  ...mods,
});
const turns: [string, ReturnType<typeof key>, boolean, boolean][] = [
  ["R on the canvas, reroute tool armed", key("r", canvas), true, true],
  ["Shift+R on the canvas, reroute tool armed", key("R", canvas), true, true],
  ["R typed in a field, reroute tool armed", key("r", field), true, false],
  ["R with no reroute tool armed", key("r", canvas), false, false],
  ["Ctrl+R (reload), reroute tool armed", key("r", canvas, { ctrlKey: true }), true, false],
  ["Cmd+R, reroute tool armed", key("r", canvas, { metaKey: true }), true, false],
  ["another key, reroute tool armed", key("t", canvas), true, false],
];
for (const [name, event, armed, turned] of turns) {
  test(`${name}: ${turned ? "turns the reroute" : "turns nothing"}`, () => {
    assert.equal(turnsArmedReroute(event, armed), turned);
  });
}
