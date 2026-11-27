// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createDisplayGraphReuser,
  reuseDisplayGraph,
} from "../src/features/recipe-studio/utils/graph/reuse-display-graph.ts";

type AnyGraph = Parameters<typeof reuseDisplayGraph>[1];

function graph(
  positions: Record<string, number>,
  extra: Record<string, unknown> = {},
): AnyGraph {
  const nodes = Object.entries(positions).map(([id, x]) => ({
    id,
    type: "builder",
    position: { x, y: 0 },
    data: { name: id, runtimeState: "idle", executionLocked: false, ...extra },
    style: { width: 400 },
  }));
  const edges = [
    {
      id: "e-a-b",
      source: "a",
      target: "b",
      type: "canvas",
      data: { path: "smoothstep", active: false },
    },
  ];
  return { nodes, edges } as unknown as AnyGraph;
}

test("unchanged nodes and edges keep their previous objects", () => {
  const reuse = createDisplayGraphReuser();
  const first = reuse(graph({ a: 0, b: 10, c: 20 }));
  const second = reuse(graph({ a: 0, b: 15, c: 20 }));
  assert.equal(second.nodes[0], first.nodes[0]);
  assert.notEqual(second.nodes[1], first.nodes[1]);
  assert.deepEqual(second.nodes[1].position, { x: 15, y: 0 });
  assert.equal(second.nodes[2], first.nodes[2]);
  assert.equal(second.edges[0], first.edges[0]);
  assert.equal(second.edges, first.edges);
});

test("an identical graph returns the previous graph", () => {
  const reuse = createDisplayGraphReuser();
  const first = reuse(graph({ a: 0, b: 10 }));
  assert.equal(reuse(graph({ a: 0, b: 10 })), first);
});

test("a changed data field, added key or new order is not reused", () => {
  const prev = graph({ a: 0, b: 10 });
  const running = reuseDisplayGraph(
    prev,
    graph({ a: 0, b: 10 }, { runtimeState: "running" }),
  );
  assert.notEqual(running.nodes[0], prev.nodes[0]);
  const extraKey = reuseDisplayGraph(
    prev,
    graph({ a: 0, b: 10 }, { note: undefined }),
  );
  assert.notEqual(extraKey.nodes[0], prev.nodes[0]);
  const reordered = graph({ b: 10, a: 0 });
  const out = reuseDisplayGraph(prev, reordered);
  assert.equal(out.nodes[0], prev.nodes[1]);
  assert.equal(out.nodes[1], prev.nodes[0]);
  assert.notEqual(out.nodes, prev.nodes);
});

test("nested objects compare one level deep by identity", () => {
  const prev = graph({ a: 0 });
  const next = graph({ a: 0 });
  (next.nodes[0].data as Record<string, unknown>).values = ["x"];
  (prev.nodes[0].data as Record<string, unknown>).values = ["x"];
  assert.notEqual(reuseDisplayGraph(prev, next).nodes[0], prev.nodes[0]);
});
