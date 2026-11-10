// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Routing passes per pan and per drag, counted through the scheduler the canvas uses.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { WireScheduler, requestRouting } = await import(
  "../src/features/flows/editor/wire-schedule.ts"
);
const { routingVersion } = await import("../src/features/flows/editor/wire-obstacles.ts");
const { routeWires, wireGridStats } = await import("../src/features/flows/editor/wire-router.ts");

type Node = {
  id: string;
  type: string;
  position: { x: number; y: number };
  measured: { width: number; height: number };
  hidden: boolean;
  selected?: boolean;
  internals: { positionAbsolute: { x: number; y: number } };
};

const reads = { position: 0 };
function card(id: string, x: number, y: number, selected = false): Node {
  const node: Node = {
    id,
    type: "flowNode",
    position: { x, y },
    measured: { width: 220, height: 120 },
    hidden: false,
    selected,
    internals: { positionAbsolute: { x, y } },
  };
  // Counts every read of a card's position: the per-update cost of noticing a change.
  return new Proxy(node, {
    get(target, key, receiver) {
      if (key === "position") reads.position += 1;
      return Reflect.get(target, key, receiver);
    },
  });
}

function graph(count: number) {
  const lookup = new Map<string, Node>();
  for (let i = 0; i < count; i += 1)
    lookup.set(`n${i}`, card(`n${i}`, (i % 6) * 400, Math.floor(i / 6) * 300));
  const edges = () =>
    Array.from({ length: count - 1 }, (_, i) => ({
      id: `e${i}`,
      source: `n${i}`,
      target: `n${i + 1}`,
      sourceHandle: "out:text",
      targetHandle: "in:value",
      data: {},
    }));
  return { lookup, edges };
}

// Stands in for the canvas: a store, React's effect deps, and one animation frame queue.
function canvas(count: number) {
  const { lookup, edges } = graph(count);
  const state = { nodeLookup: lookup, edges: edges() };
  const frames: (() => void)[] = [];
  const scheduler = new WireScheduler({
    route: (request, done) => {
      const result = routeWires(request.rects, request.edges, { hopRadius: 4 }, undefined, request.zones);
      done({ routes: [...result.routes.values()], partial: request.partial });
    },
    nextFrame: (run) => {
      frames.push(run);
      return true;
    },
    apply: () => {},
  });
  const versionOf = routingVersion();
  let deps: unknown[] = [];
  let moving: Set<string> | null = null;
  const update = () => {
    const next = [versionOf(state.nodeLookup.values()), state.edges, moving];
    if (next.some((value, i) => value !== deps[i]))
      requestRouting(scheduler, () => state as never, "flow", moving, "bus");
    deps = next;
  };
  const frame = () => {
    for (const run of frames.splice(0)) run();
  };
  const moveTo = (id: string, x: number, y: number) =>
    state.nodeLookup.set(id, card(id, x, y, true));
  return {
    state,
    scheduler,
    update,
    frame,
    moveTo,
    edges,
    setMoving: (ids: string[] | null) => {
      moving = ids ? new Set(ids) : null;
    },
  };
}

test("a pan reads no card and runs no routing pass", () => {
  const c = canvas(30);
  c.update();
  assert.deepEqual(c.scheduler.stats, { full: 1, partial: 0, skipped: 0 });
  reads.position = 0;
  // A pan only moves the viewport: the store updates, the lookup's cards stay the same objects.
  for (let k = 0; k < 60; k += 1) c.update();
  assert.equal(reads.position, 0);
  assert.deepEqual(c.scheduler.stats, { full: 1, partial: 0, skipped: 0 });
});

test("a drag runs one windowed pass per frame and one full pass on release", () => {
  const c = canvas(30);
  c.update();
  // Pressing a card selects it: new card objects and re-derived edges, the same routes.
  c.state.nodeLookup.set("n7", card("n7", 400, 300, true));
  c.state.edges = c.edges();
  c.update();
  assert.deepEqual(c.scheduler.stats, { full: 1, partial: 0, skipped: 1 });
  c.setMoving(["n7"]);
  for (let k = 1; k <= 20; k += 1) {
    c.moveTo("n7", 400 + k * 5, 300 + k * 3);
    c.update();
    c.update();
    c.frame();
  }
  assert.equal(c.scheduler.stats.partial, 20);
  // Release, then the drop's doc commit re-derives the same edges.
  c.setMoving(null);
  c.update();
  c.state.edges = c.edges();
  c.update();
  assert.deepEqual(c.scheduler.stats, { full: 2, partial: 20, skipped: 2 });
});

test("a full pass after a drag's partial passes is never skipped", () => {
  const c = canvas(12);
  c.update();
  c.setMoving(["n3"]);
  c.moveTo("n3", 900, 900);
  c.update();
  c.frame();
  // Dropped back where it started: the same cards as the first full pass, but the routes moved.
  c.moveTo("n3", 1200, 0);
  c.setMoving(null);
  c.update();
  assert.deepEqual(c.scheduler.stats, { full: 2, partial: 1, skipped: 0 });
});

test("routing passes reuse one set of grid buffers", () => {
  const { lookup, edges } = graph(30);
  const rects = [...lookup.values()].map((n) => ({ id: n.id, x: n.position.x, y: n.position.y, width: 220, height: 120 }));
  const wires = edges().map((e) => ({ id: e.id, source: e.source, target: e.target, sourceSide: "right" as const, targetSide: "left" as const, net: e.source }));
  routeWires(rects, wires);
  const before = { ...wireGridStats };
  for (let k = 0; k < 5; k += 1) routeWires(rects, wires);
  assert.ok(wireGridStats.built > before.built);
  assert.equal(wireGridStats.allocated, before.allocated);
});

test("no routing pass until every visible card has its size; a hidden one does not hold it up", () => {
  const c = canvas(6);
  const unmeasured = (id: string, hidden: boolean) => {
    const n = c.state.nodeLookup.get(id) as Node;
    c.state.nodeLookup.set(id, { ...n, hidden, measured: { width: 0, height: 0 } });
  };
  unmeasured("n2", false);
  c.update();
  assert.deepEqual(c.scheduler.stats, { full: 0, partial: 0, skipped: 0 });
  unmeasured("n2", true);
  c.update();
  assert.deepEqual(c.scheduler.stats, { full: 1, partial: 0, skipped: 0 });
});

test("dragging a note or a frame on its own sends no pass, not even a partial one", () => {
  const c = canvas(6);
  c.update();
  for (const id of ["note:n1", "group:g1"]) {
    c.setMoving([id]);
    c.update();
    c.frame();
  }
  assert.deepEqual(c.scheduler.stats, { full: 1, partial: 0, skipped: 0 });
});
