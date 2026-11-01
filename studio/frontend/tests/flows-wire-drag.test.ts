// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Dragging a card re-routes its own wires live, once per frame; a note moves no wire.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { wiresToReroute, stretchToHandles, wireMidpoint } = await import(
  "../src/features/flows/editor/wire-live.ts"
);

const src = (path: string) =>
  readFileSync(new URL(`../src/features/flows/editor/${path}`, import.meta.url), "utf8");
const e = (id: string, source: string, target: string, sh = "out:text", th = "in:value") =>
  ({ id, source, target, sourceHandle: sh, targetHandle: th }) as never;

const edges = [
  e("ab", "a", "b"),
  e("ac", "a", "c"), // shares a's output lane with ab
  e("dc", "d", "c", "out:text", "in:other"),
  e("xy", "x", "y"),
];

test("a dragged card re-routes its own wires and the ones sharing their lane, never every wire", () => {
  const ids = (moving: string[]) => wiresToReroute(edges, new Set(moving)).map((w: { id: string }) => w.id).sort();
  assert.deepEqual(ids(["b"]), ["ab", "ac"]);
  assert.deepEqual(ids(["x"]), ["xy"]);
  assert.deepEqual(ids(["note:n1"]), [], "a note moves no wire");
  assert.deepEqual(ids(["group:g1"]), [], "a frame on its own moves no wire");
});

test("the controller sends partial requests at most once per frame and a full pass on release", () => {
  const routing = src("wire-routing.tsx");
  assert.match(routing, /if \(!live\.length\) return;/);
  assert.match(routing, /partial: true,[\s\S]*pumpNextFrame\(\);/);
  assert.match(routing, /frame = requestAnimationFrame\(/);
  assert.match(routing, /const next = partial\s*\? new Map<string, WireRoute>\(previous\)/);
  assert.match(src("canvas.tsx"), /<WireRouterController[^>]*moving=\{moving\}/);
  assert.match(routing, /show=\{deletable !== false && \(hover \|\| !!selected\)\}/);
});

test("a route a frame behind its card follows the handles instead of jumping to a curve", () => {
  const route = [{ x: 260, y: 60 }, { x: 300, y: 60 }, { x: 300, y: 200 }, { x: 500, y: 200 }];
  assert.deepEqual(stretchToHandles(route, 262, 70, 500, 205), [
    { x: 262, y: 70 }, { x: 300, y: 70 }, { x: 300, y: 205 }, { x: 500, y: 205 },
  ]);
  assert.equal(stretchToHandles(route, 320, 70, 500, 200), null, "would fold back");
});

test("each wire of a fan-in has its own × point before the shared leg", () => {
  const one = wireMidpoint([{ x: 0, y: 0 }, { x: 200, y: 0 }, { x: 200, y: 100 }, { x: 240, y: 100 }]);
  const two = wireMidpoint([{ x: 0, y: 200 }, { x: 200, y: 200 }, { x: 200, y: 100 }, { x: 240, y: 100 }]);
  assert.notDeepEqual(one, two);
  assert.ok(one.x < 200 && two.x <= 200);
});

test("a reroute's wire segments have no draggable end; they move with the reroute", () => {
  assert.match(src("derive-rf.ts"), /reconnectable: false,\s*data: \{\s*kind: "data",\s*portType: info\.portType,\s*ends: info\.ends,/);
  assert.match(src("use-wire-reconnect.ts"), /isWireSegment\(before\.graph, old\.id\)/);
});
