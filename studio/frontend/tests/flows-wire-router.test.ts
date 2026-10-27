// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { routeWires, buildWirePath, wirePortPoint } = await import(
  "../src/features/flows/editor/wire-router.ts"
);

type Rect = Parameters<typeof routeWires>[0][number];
type EdgeIn = Parameters<typeof routeWires>[1][number];
type Route = ReturnType<typeof routeWires>["routes"] extends Map<string, infer R> ? R : never;
type Point = { x: number; y: number };

const card = (id: string, x: number, y: number, width = 220, height = 120): Rect => ({
  id,
  x,
  y,
  width,
  height,
});

const wire = (
  source: string,
  target: string,
  sourceSide: EdgeIn["sourceSide"] = "right",
  targetSide: EdgeIn["targetSide"] = "left",
): EdgeIn => ({
  id: `${source}->${target}`,
  source,
  target,
  sourceSide,
  targetSide,
  net: `${source}:${sourceSide}`,
});

function segments(points: Point[]): Array<[Point, Point]> {
  const out: Array<[Point, Point]> = [];
  for (let k = 0; k + 1 < points.length; k += 1) {
    out.push([points[k], points[k + 1]]);
  }
  return out;
}

function hitsInterior(a: Point, b: Point, rect: Rect): boolean {
  return (
    Math.max(a.x, b.x) > rect.x + 0.5 &&
    Math.min(a.x, b.x) < rect.x + rect.width - 0.5 &&
    Math.max(a.y, b.y) > rect.y + 0.5 &&
    Math.min(a.y, b.y) < rect.y + rect.height - 0.5
  );
}

function assertWireRules(rects: Rect[], edges: EdgeIn[], routes: Map<string, Route>): void {
  const byId = new Map(rects.map((rect) => [rect.id, rect] as const));
  for (const edge of edges) {
    const route = routes.get(edge.id);
    assert.ok(route, `${edge.id} has a route`);
    assert.equal(route.blocked, false, `${edge.id} found a clean path`);
    const points = route.points;
    for (const [a, b] of segments(points)) {
      assert.ok(
        Math.abs(a.x - b.x) < 0.01 || Math.abs(a.y - b.y) < 0.01,
        `${edge.id} is orthogonal`,
      );
      for (const rect of rects) {
        assert.ok(!hitsInterior(a, b, rect), `${edge.id} passes under ${rect.id}`);
      }
    }
    const source = byId.get(edge.source) as Rect;
    const target = byId.get(edge.target) as Rect;
    assert.deepEqual(points[0], wirePortPoint(source, edge.sourceSide));
    assert.deepEqual(points[points.length - 1], wirePortPoint(target, edge.targetSide));
    // Leaves a right/bottom output outwards and enters a top/left input inwards.
    const [p0, p1] = [points[0], points[1]];
    if (edge.sourceSide === "right") {
      assert.ok(p1.x > p0.x && p1.y === p0.y, `${edge.id} leaves rightwards`);
    } else {
      assert.ok(p1.y > p0.y && p1.x === p0.x, `${edge.id} leaves downwards`);
    }
    const [q0, q1] = [points[points.length - 2], points[points.length - 1]];
    if (edge.targetSide === "left") {
      assert.ok(q1.x > q0.x && q1.y === q0.y, `${edge.id} enters from the left`);
    } else {
      assert.ok(q1.y > q0.y && q1.x === q0.x, `${edge.id} enters from the top`);
    }
  }
}

function assertNoSharedLanes(edges: EdgeIn[], routes: Map<string, Route>): void {
  type Seg = {
    net: string;
    vertical: boolean;
    coord: number;
    lo: number;
    hi: number;
    id: string;
    port: Point;
  };
  const all: Seg[] = [];
  const ports = new Map<string, Point>();
  for (const edge of edges) {
    const points = (routes.get(edge.id) as Route).points;
    const end = points[points.length - 1];
    const key = `${edge.target}:${edge.targetSide}`;
    if (!ports.has(key)) {
      ports.set(key, end);
    }
    const port = ports.get(key) as Point;
    for (const [a, b] of segments(points)) {
      const vertical = Math.abs(a.x - b.x) < 0.01;
      all.push({
        id: edge.id,
        net: edge.net,
        vertical,
        coord: vertical ? a.x : a.y,
        lo: vertical ? Math.min(a.y, b.y) : Math.min(a.x, b.x),
        hi: vertical ? Math.max(a.y, b.y) : Math.max(a.x, b.x),
        port,
      });
    }
  }
  for (let i = 0; i < all.length; i += 1) {
    for (let j = i + 1; j < all.length; j += 1) {
      const a = all[i];
      const b = all[j];
      // One output fanning out, or several wires into one input, is a bundle, not two lanes.
      if (a.net === b.net || a.port === b.port || a.vertical !== b.vertical) {
        continue;
      }
      const sameLine = Math.abs(a.coord - b.coord) < 0.5;
      const overlap = Math.min(a.hi, b.hi) - Math.max(a.lo, b.lo);
      // Lane nudging can leave two unrelated corners a few pixels apart on one line; anything
      // longer than one lane gap would read as a shared wire.
      assert.ok(!(sameLine && overlap > 6), `${a.id} and ${b.id} share a lane`);
    }
  }
}

test("a wire goes around a card in its way instead of under it", () => {
  const rects = [card("a", 0, 0), card("blocker", 320, -20, 220, 160), card("b", 640, 0)];
  const edges = [wire("a", "b")];
  const { routes } = routeWires(rects, edges);
  assertWireRules(rects, edges, routes);
  assert.ok((routes.get("a->b") as Route).points.length > 2, "the path bends around the blocker");
});

test("outputs leave right or bottom and inputs enter top or left", () => {
  const rects = [card("a", 0, 0), card("b", 400, 300), card("c", 0, 400)];
  const edges = [wire("a", "b", "bottom", "top"), wire("a", "c", "right", "left"), wire("c", "b", "right", "top")];
  const { routes } = routeWires(rects, edges);
  assertWireRules(rects, edges, routes);
});

test("a port squeezed next to another card still finds a clean way out", () => {
  // The model card ends 28px from the LLM card's input, so the input stub sits in its clearance band.
  const rects = [card("model", 0, -100, 260, 184), card("llm", 288, 0), card("src", -600, 300)];
  const edges = [wire("src", "llm"), wire("model", "llm", "bottom", "top")];
  const { routes } = routeWires(rects, edges);
  assertWireRules(rects, edges, routes);
});

test("a card dropped over a port does not cut its wire off", () => {
  const rects = [card("a", 0, 0), card("cover", 180, 20, 220, 120), card("b", 700, 0), card("wall", 450, -200, 100, 500)];
  const edges = [wire("a", "b")];
  const { routes } = routeWires(rects, edges);
  const route = routes.get("a->b") as Route;
  assert.equal(route.blocked, false, "the wire still routes");
  for (const [a, b] of segments(route.points)) {
    for (const rect of rects.filter((r) => r.id !== "cover")) {
      assert.ok(!hitsInterior(a, b, rect), `passes under ${rect.id}`);
    }
  }
});

test("a feedback wire runs backwards around the graph without breaking the side rules", () => {
  const rects = [card("gen", 0, 0), card("judge", 400, 0), card("bench", 800, 0)];
  const edges = [wire("gen", "judge"), wire("judge", "bench"), wire("bench", "gen")];
  const { routes } = routeWires(rects, edges);
  assertWireRules(rects, edges, routes);
});

test("wires from different outputs that share a corridor get separate lanes", () => {
  const rects = [
    card("s1", 0, 0),
    card("s2", 0, 200),
    card("s3", 0, 400),
    card("t1", 700, 400),
    card("t2", 700, 200),
    card("t3", 700, 0),
  ];
  const edges = [wire("s1", "t1"), wire("s2", "t2"), wire("s3", "t3")];
  const { routes } = routeWires(rects, edges);
  assertWireRules(rects, edges, routes);
  assertNoSharedLanes(edges, routes);
});

test("a crossing draws a hop arc and never looks like a junction", () => {
  const rects = [card("top", 0, 0), card("bottom", 0, 400), card("a", 600, 0), card("b", 600, 400)];
  const edges = [wire("top", "b"), wire("bottom", "a")];
  const { routes } = routeWires(rects, edges);
  assertWireRules(rects, edges, routes);
  const hops = [...routes.values()].flatMap((route) => route.hops);
  assert.equal(hops.length, 1, "exactly one crossing gets a hop");
  assert.equal([...routes.values()].flatMap((route) => route.junctions).length, 0);
  const owner = [...routes.values()].find((route) => route.hops.length > 0) as Route;
  assert.match(buildWirePath(owner.points, owner.hops), / A 4 4 0 0 [01] /);
});

test("one output feeding two inputs shares a trunk and marks the fan-out with a junction", () => {
  const rects = [card("src", 0, 200), card("t1", 600, 0), card("t2", 600, 400)];
  const edges = [wire("src", "t1"), wire("src", "t2")];
  const { routes } = routeWires(rects, edges);
  assertWireRules(rects, edges, routes);
  const junctions = [...routes.values()].flatMap((route) => route.junctions);
  assert.equal(junctions.length, 1);
  assert.ok(junctions[0].x > 220, "the junction sits on the wire, past the port");
});

test("moving a card only reroutes the wires near it", () => {
  const rects = [card("a", 0, 0), card("b", 400, 0), card("c", 0, 600), card("d", 400, 600)];
  const edges = [wire("a", "b"), wire("c", "d")];
  const first = routeWires(rects, edges);
  assert.equal(first.rerouted, 2);
  const moved = rects.map((rect) => (rect.id === "d" ? { ...rect, y: 650 } : rect));
  const second = routeWires(moved, edges, {}, first.cache);
  assert.equal(second.rerouted, 1, "only c->d is searched again");
  assert.deepEqual(second.routes.get("a->b")?.points, first.routes.get("a->b")?.points);
  assertWireRules(moved, edges, second.routes);
});

function bigRecipe(columns: number, rows: number): { rects: Rect[]; edges: EdgeIn[] } {
  const rects: Rect[] = [];
  const edges: EdgeIn[] = [];
  for (let c = 0; c < columns; c += 1) {
    for (let r = 0; r < rows; r += 1) {
      rects.push(card(`n${c}-${r}`, c * 420, r * 220 + (c % 2) * 40, 260, 140));
    }
  }
  // Deterministic pseudo-random wiring: each card feeds 1-2 cards in later columns.
  let seed = 7;
  const rand = (): number => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  for (let c = 0; c < columns - 1; c += 1) {
    for (let r = 0; r < rows; r += 1) {
      const fan = rand() < 0.4 ? 2 : 1;
      const used = new Set<string>();
      for (let k = 0; k < fan; k += 1) {
        const tc = Math.min(columns - 1, c + 1 + Math.floor(rand() * 2));
        const tr = Math.floor(rand() * rows);
        const target = `n${tc}-${tr}`;
        if (used.has(target)) {
          continue;
        }
        used.add(target);
        edges.push(wire(`n${c}-${r}`, target));
      }
    }
  }
  return { rects, edges };
}

test("a 50-card recipe routes cleanly and fast enough to run on drop", () => {
  const { rects, edges } = bigRecipe(10, 5);
  assert.equal(rects.length, 50);
  routeWires(rects, edges); // warm up the JIT
  const runs = 5;
  const started = performance.now();
  let result = routeWires(rects, edges);
  for (let k = 1; k < runs; k += 1) {
    result = routeWires(rects, edges);
  }
  const fullMs = (performance.now() - started) / runs;

  const moved = rects.map((rect) => (rect.id === "n4-2" ? { ...rect, y: rect.y + 30 } : rect));
  const dropStarted = performance.now();
  const incremental = routeWires(moved, edges, {}, result.cache);
  const dropMs = performance.now() - dropStarted;

  console.log(
    `50 cards, ${edges.length} wires: full reroute ${fullMs.toFixed(1)} ms, ` +
      `reroute after one drop ${dropMs.toFixed(1)} ms (${incremental.rerouted} wires searched)`,
  );
  assertWireRules(rects, edges, result.routes);
  assertNoSharedLanes(edges, result.routes);
  assertWireRules(moved, edges, incremental.routes);
  assert.ok(fullMs < 1000, `full reroute took ${fullMs} ms`);
});

test("a wire stays out of a group frame it has no end in, and runs freely inside its own", () => {
  const rects = [card("a", 0, 0), card("b", 900, 0), card("in1", 360, 260), card("in2", 640, 260)];
  const frame = { id: "g", x: 320, y: -120, width: 560, height: 560 };
  const edges = [wire("a", "b"), wire("in1", "in2")];
  const { routes } = routeWires(rects, edges, {}, undefined, [frame]);
  assertWireRules(rects, edges, routes);
  for (const [a, b] of segments((routes.get("a->b") as Route).points)) {
    assert.ok(!hitsInterior(a, b, frame), "a->b goes around the frame");
  }
  assert.equal((routes.get("in1->in2") as Route).points.length, 2, "in1->in2 stays straight inside");
});

test("a wire leaving a group crosses only its own frame", () => {
  const rects = [card("inner", 360, 100), card("out", 1100, 100)];
  const own = { id: "own", x: 320, y: -40, width: 400, height: 400 };
  const other = { id: "other", x: 760, y: -200, width: 280, height: 800 };
  const edges = [wire("inner", "out")];
  const { routes } = routeWires(rects, edges, {}, undefined, [own, other]);
  const route = routes.get("inner->out") as Route;
  assert.equal(route.blocked, false);
  for (const [a, b] of segments(route.points)) {
    assert.ok(!hitsInterior(a, b, other), "goes around the other frame");
  }
});

test("past the cell budget each wire routes on a local grid and still avoids cards", () => {
  const { rects, edges } = bigRecipe(10, 5);
  const local = routeWires(rects, edges, { cellBudget: 1 });
  assertWireRules(rects, edges, local.routes);
  assert.equal(local.rerouted, edges.length);
});

test("the path rounds every bend and keeps hops on the straight part", () => {
  const points = [
    { x: 0, y: 0 },
    { x: 100, y: 0 },
    { x: 100, y: 60 },
    { x: 200, y: 60 },
  ];
  const square = buildWirePath(points, [], 4, 0);
  assert.equal(square, "M 0 0 L 100 0 L 100 60 L 200 60");
  const round = buildWirePath(points, [{ x: 50, y: 0 }], 4, 10);
  assert.equal(
    round,
    "M 0 0 L 46 0 A 4 4 0 0 1 54 0 L 90 0 Q 100 0 100 10 L 100 50 Q 100 60 110 60 L 200 60",
  );
  // A short leg caps the radius at half its length.
  const tight = buildWirePath(
    [
      { x: 0, y: 0 },
      { x: 50, y: 0 },
      { x: 50, y: 8 },
      { x: 100, y: 8 },
    ],
    [],
    4,
    10,
  );
  assert.match(tight, /Q 50 0 50 4 L 50 4 Q 50 8 54 8/);
});

test("Comic Strip Machine: wires into a loop frame take the short way, not a lap round the graph", () => {
  // Card and frame geometry as the editor measures the bundled example.
  const rects = [
    card("look", 640, 0, 260, 97),
    card("captions", 640, 380, 260, 97),
    card("ref", 960, 0, 260, 249),
    card("each", 1024, 304, 260, 124),
    card("draw", 1344, 304, 260, 249),
    card("page", 1792, 190, 260, 169),
  ];
  const frame = { id: "loop", x: 988, y: 172, width: 716, height: 316 };
  const at = (edge: EdgeIn, sourceOffset: number, targetOffset: number): EdgeIn => ({
    ...edge,
    sourceOffset,
    targetOffset,
    sink: `${edge.target}:${targetOffset}`,
  });
  const edges = [
    at(wire("ref", "draw"), 57, 105),
    at(wire("each", "page"), 105, 57),
    at(wire("captions", "page"), 57, 81),
    at(wire("draw", "each"), 57, 81),
  ];
  const { routes } = routeWires(rects, edges, {}, undefined, [frame]);
  const length = (id: string): number =>
    segments((routes.get(id) as Route).points).reduce(
      (sum, [a, b]) => sum + Math.abs(a.x - b.x) + Math.abs(a.y - b.y),
      0,
    );
  assert.ok(length("ref->draw") < 600, `ref->draw is ${length("ref->draw")} px`);
  assert.ok(length("each->page") < 800, `each->page is ${length("each->page")} px`);
  for (const edge of edges) {
    const route = routes.get(edge.id) as Route;
    assert.equal(route.blocked, false);
    for (const [a, b] of segments(route.points)) {
      for (const rect of rects) {
        assert.ok(!hitsInterior(a, b, rect), `${edge.id} passes under ${rect.id}`);
      }
    }
  }
  for (const [a, b] of segments((routes.get("captions->page") as Route).points)) {
    assert.ok(!hitsInterior(a, b, frame), "captions->page goes round the loop frame");
  }
});
