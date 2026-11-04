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
  const rects = [card("a", 0, 0), card("b", 940, 0), card("in1", 360, 260), card("in2", 640, 260)];
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

function gapTo(a: Point, b: Point, rect: Rect): number {
  const dx = Math.max(rect.x - Math.max(a.x, b.x), Math.min(a.x, b.x) - (rect.x + rect.width), 0);
  const dy = Math.max(rect.y - Math.max(a.y, b.y), Math.min(a.y, b.y) - (rect.y + rect.height), 0);
  return Math.max(dx, dy);
}

test("no wire runs closer than the hard clearance to a card, its own included past the port leg", () => {
  const hard = 18;
  const dense = bigRecipe(10, 5);
  const tight = {
    // Columns 100 px apart and rows 60 px apart, like a cramped story graph.
    rects: [
      card("a", 0, 0, 220, 160),
      card("b", 320, 80, 220, 160),
      card("c", 320, 300, 220, 120),
      card("d", 640, 0, 220, 160),
      card("e", 640, 220, 220, 160),
    ],
    edges: [wire("a", "d"), wire("a", "e"), wire("b", "e"), wire("c", "d"), wire("a", "c")],
  };
  for (const { rects, edges } of [dense, tight]) {
    const { routes } = routeWires(rects, edges, { hardClearance: hard });
    for (const edge of edges) {
      const points = (routes.get(edge.id) as Route).points;
      segments(points).forEach(([a, b], k, all) => {
        const portLeg = k === 0 || k === all.length - 1;
        for (const rect of rects) {
          if (portLeg && (rect.id === edge.source || rect.id === edge.target)) {
            continue;
          }
          assert.ok(
            gapTo(a, b, rect) >= hard - 0.5,
            `${edge.id} runs ${gapTo(a, b, rect)} px from ${rect.id}`,
          );
        }
      });
      // The port legs go straight out at least the clearance before turning.
      const legs = segments(points);
      const [first, last] = [legs[0], legs[legs.length - 1]];
      assert.ok(Math.abs(first[1].x - first[0].x) >= hard, `${edge.id} turns too close to its output`);
      assert.ok(Math.abs(last[1].x - last[0].x) >= hard, `${edge.id} turns too close to its input`);
    }
  }
});

test("wires crossing the same gap share it as parallel lanes, centred between the columns", () => {
  const rects = [
    card("s1", 0, 0, 220, 100),
    card("s2", 0, 140, 220, 100),
    card("s3", 0, 280, 220, 100),
    card("t1", 420, 600, 220, 100),
    card("t2", 420, 740, 220, 100),
    card("t3", 420, 880, 220, 100),
  ];
  const edges = [wire("s1", "t1"), wire("s2", "t2"), wire("s3", "t3")];
  const { routes } = routeWires(rects, edges);
  assertWireRules(rects, edges, routes);
  const runs = edges.map((edge) => {
    const vertical = segments((routes.get(edge.id) as Route).points).find(
      ([a, b]) => Math.abs(a.x - b.x) < 0.01 && a.x > 220 && a.x < 420,
    );
    assert.ok(vertical, `${edge.id} runs down the gap`);
    return (vertical as [Point, Point])[0].x;
  });
  const xs = [...runs].sort((a, b) => a - b);
  assert.equal(new Set(xs).size, 3, "three separate lanes");
  for (let k = 1; k < xs.length; k += 1) {
    assert.ok(Math.abs(xs[k] - xs[k - 1] - 9) < 0.6, `lane pitch ${xs[k] - xs[k - 1]}`);
  }
  const mid = (xs[0] + xs[2]) / 2;
  assert.ok(Math.abs(mid - 320) <= 12, `bundle centred near the gap's middle (at ${mid})`);
});

function assertFanIn(edges: EdgeIn[], routes: Map<string, Route>, target: Rect): void {
  const port = wirePortPoint(target, "left");
  const merge = { x: port.x - 24, y: port.y };
  const into = edges.filter((edge) => edge.target === target.id).map((edge) => routes.get(edge.id) as Route);
  for (const route of into) {
    for (const hop of route.hops) {
      const onFanIn = into.some(
        (other) =>
          other !== route &&
          segments(other.points).some(
            ([a, b]) => Math.abs(a.x - hop.x) < 0.5 && Math.abs(b.x - hop.x) < 0.5,
          ),
      );
      assert.ok(!onFanIn, `${route.id} hops a wire into the same port`);
    }
    const [q0, q1] = route.points.slice(-2);
    assert.deepEqual(q1, port);
    // One shared leg in: every wire is on the port's line by the merge point (a wire already on
    // that line just runs straight through it).
    assert.ok(q0.y === port.y && q0.x <= merge.x, `${route.id} joins at the merge point, then one leg in`);
    for (const [a, b] of segments(route.points.slice(0, -1))) {
      assert.ok(Math.max(a.x, b.x) <= merge.x + 0.01, `${route.id} squeezes in past the merge point`);
    }
  }
  const dots = new Set(into.flatMap((route) => route.junctions).map((p) => `${p.x}:${p.y}`));
  assert.deepEqual([...dots], [`${merge.x}:${merge.y}`], "one junction, at the merge point");
}

test("wires into one input join one lane a fixed distance out, with one junction and no hops", () => {
  // Poster studio's Make List: one wire straight in, one from above-left, one from below-left.
  const rects = [
    card("image", 0, 0, 240, 160),
    card("make", 420, 0, 240, 160),
    card("above", -500, -400),
    card("below", -500, 500),
  ];
  const edges = [wire("image", "make"), wire("above", "make"), wire("below", "make")];
  const { routes } = routeWires(rects, edges);
  assertWireRules(rects, edges, routes);
  assertFanIn(edges, routes, rects[1]);
});

test("a wire into a different input still hops the fan-in it crosses", () => {
  const rects = [
    card("image", 0, 0, 240, 160),
    card("make", 420, 0, 240, 160),
    card("above", -500, -400),
    card("below", -500, 500),
    card("cross", -500, 250),
    card("other", 900, -400),
  ];
  const edges = [
    wire("image", "make"),
    wire("above", "make"),
    wire("below", "make"),
    wire("cross", "other"),
  ];
  const { routes } = routeWires(rects, edges);
  assertWireRules(rects, edges, routes);
  assertFanIn(edges, routes, rects[1]);
  const crossing = routes.get("cross->other") as Route;
  const fanIn = edges.filter((e) => e.target === "make").map((e) => routes.get(e.id) as Route);
  assert.ok(
    crossing.hops.length + fanIn.reduce((n, route) => n + route.hops.length, 0) > 0,
    "the unrelated crossing still gets a hop",
  );
});

test("unrelated wires never meet end to end on one line, so neither reads as the other's start", () => {
  // Dataset Forge: Checkpoint fans out to Chat and Export Model, and a Text card under Fine-tune
  // feeds Chat's prompt from the same height as Export Model's input.
  const rects = [
    card("tune", 1280, 190, 260, 200),
    card("ask", 1280, 418, 260, 120),
    card("test", 1600, 190, 260, 180),
    card("gguf", 1920, 418, 260, 160),
  ];
  const at = (source: string, target: string, sourceOffset: number, targetOffset: number): EdgeIn => ({
    ...wire(source, target),
    sourceOffset,
    targetOffset,
    net: `${source}:out`,
    sink: `${target}:${targetOffset}`,
  });
  const edges = [at("tune", "test", 60, 50), at("tune", "gguf", 60, 60), at("ask", "test", 60, 80)];
  const { routes } = routeWires(rects, edges);
  assert.ok([...routes.values()].every((route) => !route.blocked));
  const runs = (id: string) => {
    const points = (routes.get(id) as Route).points;
    return points.slice(1).flatMap((b, k) => {
      const a = points[k];
      return Math.abs(a.y - b.y) < 0.01 && Math.abs(a.x - b.x) > 0.01
        ? [{ y: a.y, lo: Math.min(a.x, b.x), hi: Math.max(a.x, b.x) }]
        : [];
    });
  };
  for (const a of runs("ask->test"))
    for (const b of runs("tune->gguf"))
      assert.ok(
        Math.abs(a.y - b.y) > 0.5 || Math.max(a.lo, b.lo) - Math.min(a.hi, b.hi) > 27,
        `runs meet on y=${a.y}: ${JSON.stringify([a, b])}`,
      );
});

// An image card feeding two Preview cards in a row, plus a trunk from a card up-left into the first.
function previewRow(rows: Array<[string, number, number, number]>, images = true) {
  const rects = rows.map(([id, x, y, height]) => card(id, x, y, 260, height));
  const port = (source: string, target: string, handle: string, from: number): EdgeIn => ({
    id: `${source}:${handle}->${target}`,
    source,
    target,
    sourceSide: "right",
    targetSide: "left",
    sourceOffset: from,
    targetOffset: 65,
    net: `${source}:${handle}`,
    sink: `${target}:value`,
  });
  const edges = [
    port("gen", "mid", "image", 65),
    port("gen", "right", "image", 65),
    port("top", "mid", "text", 65),
    ...(images ? [port("gen", "right", "images", 85)] : []),
  ];
  return { rects, edges, routes: routeWires(rects, edges, { hopRadius: 4 }).routes };
}

test("a wire into the next card in a row goes around the card between, not under it", () => {
  // The right card's input stub lands in the middle card's clearance band (40px gap).
  const { rects, edges, routes } = previewRow([
    ["top", -345, -296, 97],
    ["gen", 0, 0, 363],
    ["mid", 316, 34, 340],
    ["right", 616, 53, 211],
  ]);
  const mid = rects.find((rect) => rect.id === "mid") as Rect;
  for (const edge of edges) {
    const route = routes.get(edge.id) as Route;
    assert.equal(route.blocked, false, `${edge.id} routes`);
    for (const [a, b] of segments(route.points)) {
      assert.ok(edge.target === "mid" || !hitsInterior(a, b, mid), `${edge.id} passes under mid`);
    }
  }
});

function assertCleanHops(routes: Map<string, Route>, radius = 4): void {
  const all = [...routes.values()];
  const junctions = all.flatMap((route) => route.junctions);
  for (const route of all) {
    const hops = [...route.hops].sort((a, b) => a.y - b.y || a.x - b.x);
    for (let k = 1; k < hops.length; k += 1) {
      assert.ok(
        hops[k].y !== hops[k - 1].y || hops[k].x - hops[k - 1].x > radius * 2,
        `${route.id} stacks two hops at ${hops[k].x},${hops[k].y}`,
      );
    }
    for (const hop of hops) {
      assert.ok(
        !junctions.some((j) => Math.abs(j.x - hop.x) <= radius * 2 && Math.abs(j.y - hop.y) <= radius),
        `${route.id} hops over a junction dot at ${hop.x},${hop.y}`,
      );
      // The crossed wire runs well past the arc on both sides: a real crossing, not a lane jog.
      const crossed = all.some(
        (other) =>
          other !== route &&
          segments(other.points).some(
            ([a, b]) =>
              Math.abs(a.x - hop.x) < 0.01 &&
              Math.abs(b.x - hop.x) < 0.01 &&
              Math.min(a.y, b.y) < hop.y - radius &&
              Math.max(a.y, b.y) > hop.y + radius,
          ),
      );
      assert.ok(crossed, `${route.id} hops at ${hop.x},${hop.y} where nothing crosses it`);
    }
  }
}

test("a hop over wires that merge into one input is drawn once", () => {
  const { routes } = previewRow([
    ["top", -188, -158, 85],
    ["gen", 0, 0, 240],
    ["mid", 306, 59, 235],
    ["right", 650, 61, 348],
  ]);
  assertCleanHops(routes);
  for (const route of routes.values()) {
    const path = buildWirePath(route.points, route.hops, 4, 10);
    // A hop inside a rounded corner is left out, so at most one arc per hop.
    assert.ok((path.match(/ A /g)?.length ?? 0) <= route.hops.length, `${route.id} draws each hop once`);
  }
});

test("no hop where a wire forks or joins right on the line it would hop", () => {
  const { routes } = previewRow([
    ["top", -377, -227, 93],
    ["gen", 0, 0, 277],
    ["mid", 353, 11, 335],
    ["right", 793, -20, 343],
  ]);
  assertCleanHops(routes);
});

