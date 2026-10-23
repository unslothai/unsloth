// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { HANDLE_IDS } = await import("../src/features/recipe-studio/utils/handles.ts");
const { getRecipeNodePorts, resolveRecipeEdgeHandles } = await import(
  "../src/features/recipe-studio/utils/ports.ts"
);
const { deriveDisplayGraph } = await import(
  "../src/features/recipe-studio/utils/graph/derive-display-graph.ts"
);
const {
  expandRerouteEdges,
  findRerouteUpstream,
  insertRerouteIntoEdge,
  logicalEdgeId,
  removeReroutesFromEdges,
} = await import("../src/features/recipe-studio/utils/graph/reroutes.ts");
const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);
const { buildRecipePayload, withoutReroutes } = await import(
  "../src/features/recipe-studio/utils/payload/build-payload.ts"
);
const { findWireNear, routeWires, wireEmphasis } = await import(
  "../src/features/recipe-studio/utils/graph/wire-router.ts"
);

const recipeDir = new URL("../src/features/data-recipes/learning-recipes/", import.meta.url);
const learningRecipes = readdirSync(recipeDir)
  .filter((name) => name.endsWith(".json"))
  .map((name) => ({
    name,
    payload: JSON.parse(readSrc(`features/data-recipes/learning-recipes/${name}`)),
  }));

function importSnapshot(payload: unknown) {
  const result = importRecipePayload(JSON.stringify(payload));
  assert.deepEqual(result.errors, []);
  assert.ok(result.snapshot);
  return result.snapshot;
}

function display(snapshot: ReturnType<typeof importSnapshot>, aux = false) {
  return deriveDisplayGraph({
    nodes: snapshot.nodes,
    edges: snapshot.edges,
    configs: snapshot.configs,
    auxNodePositions: {},
    reroutes: snapshot.reroutes ?? {},
    llmAuxVisibility: aux
      ? Object.fromEntries(Object.keys(snapshot.configs).map((id) => [id, true]))
      : {},
  });
}

test("every card puts its inputs on the left, outputs on the right, each with a label", () => {
  for (const { name, payload } of learningRecipes) {
    const graph = display(importSnapshot(payload), true);
    for (const node of graph.nodes) {
      const ports = (node.data as { ports?: { id: string; side: string; label: string }[] }).ports;
      for (const port of ports ?? []) {
        const isInput = /-in($|-)/.test(port.id) && !port.id.startsWith("llm-input");
        assert.equal(port.side, isInput ? "left" : "right", `${name} ${node.id} ${port.id}`);
        if (node.type !== "reroute") {
          assert.ok(port.label.trim(), `${name} ${node.id} ${port.id} has a label`);
        }
      }
    }
  }
});

test("every wire in every learning recipe lands on a port its two cards show", () => {
  for (const { name, payload } of learningRecipes) {
    const graph = display(importSnapshot(payload), true);
    const ports = new Map(
      graph.nodes.map((node) => [
        node.id,
        new Set(((node.data as { ports?: { id: string }[] }).ports ?? []).map((p) => p.id)),
      ]),
    );
    assert.ok(graph.edges.length > 0, name);
    for (const edge of graph.edges) {
      assert.ok(ports.get(edge.source)?.has(edge.sourceHandle ?? ""), `${name}: ${edge.id} out`);
      assert.ok(ports.get(edge.target)?.has(edge.targetHandle ?? ""), `${name}: ${edge.id} in`);
    }
  }
});

test("model, tool and provider wires each get their own labelled input", () => {
  const llm = { id: "l", kind: "llm", name: "reply", llm_type: "text", tool_alias: "" };
  const model = { id: "m", kind: "model_config", name: "writer" };
  const tool = { id: "t", kind: "tool_config", name: "docs" };
  const provider = { id: "p", kind: "model_provider", name: "local" };
  const column = { id: "c", kind: "sampler", name: "topic" };
  type Config = Parameters<typeof resolveRecipeEdgeHandles>[0];
  const handles = (source: unknown, target: unknown) =>
    resolveRecipeEdgeHandles(source as Config, target as Config);
  assert.deepEqual(handles(model, llm), {
    sourceHandle: HANDLE_IDS.semanticOut,
    targetHandle: HANDLE_IDS.modelIn,
  });
  assert.deepEqual(handles(tool, llm), {
    sourceHandle: HANDLE_IDS.semanticOut,
    targetHandle: HANDLE_IDS.toolsIn,
  });
  assert.deepEqual(handles(provider, model), {
    sourceHandle: HANDLE_IDS.semanticOut,
    targetHandle: HANDLE_IDS.semanticIn,
  });
  assert.deepEqual(handles(column, llm), {
    sourceHandle: HANDLE_IDS.dataOut,
    targetHandle: HANDLE_IDS.dataIn,
  });
  const labels = getRecipeNodePorts(llm as Config, { toolsInput: true }).map(
    (port) => `${port.side}:${port.kind}:${port.label}`,
  );
  assert.deepEqual(labels, [
    "left:data:prompt refs",
    "left:model:model",
    "left:tool:tools",
    "right:data:reply",
  ]);
});

test("a recipe saved top to bottom opens left to right with every edge", () => {
  const sql = learningRecipes.find((recipe) => recipe.name === "text-to-sql.json");
  assert.ok(sql);
  const tb = structuredClone(sql.payload);
  tb.ui.layout_direction = "TB";
  const snapshot = importSnapshot(tb);
  assert.equal(snapshot.layoutDirection, "LR");
  assert.equal(snapshot.edges.length, importSnapshot(sql.payload).edges.length);
});

const edge = (id: string, source: string, target: string, via?: string[]) => ({
  id,
  source,
  target,
  sourceHandle: HANDLE_IDS.dataOut,
  targetHandle: HANDLE_IDS.dataIn,
  ...(via ? { data: { via } } : {}),
});

test("a wire through reroutes is drawn as one segment per hop and maps back to its edge", () => {
  const reroutes = { r1: { x: 0, y: 0 }, r2: { x: 50, y: 0 } };
  const segments = expandRerouteEdges([edge("e", "a", "b", ["r1", "r2"])], reroutes);
  assert.deepEqual(
    segments.map((s) => [s.source, s.sourceHandle, s.target, s.targetHandle]),
    [
      ["a", HANDLE_IDS.dataOut, "r1", HANDLE_IDS.rerouteIn],
      ["r1", HANDLE_IDS.rerouteOut, "r2", HANDLE_IDS.rerouteIn],
      ["r2", HANDLE_IDS.rerouteOut, "b", HANDLE_IDS.dataIn],
    ],
  );
  assert.deepEqual(new Set(segments.map((s) => logicalEdgeId(s.id))), new Set(["e"]));
  // A reroute that no longer exists is skipped rather than leaving a dangling segment.
  assert.equal(expandRerouteEdges([edge("e", "a", "b", ["gone"])], reroutes).length, 1);
});

test("dropping, fanning out from and deleting a reroute keep the recipe edges intact", () => {
  let edges: ReturnType<typeof removeReroutesFromEdges> = [edge("e", "a", "b")];
  edges = insertRerouteIntoEdge(edges, "e", "r1");
  assert.deepEqual(edges[0].data, { via: ["r1"] });
  // Splitting the second segment of a -> r1 -> b puts r2 after r1.
  edges = insertRerouteIntoEdge(edges, "e~1", "r2");
  assert.deepEqual(edges[0].data, { via: ["r1", "r2"] });
  const upstream = findRerouteUpstream(edges, "r2");
  assert.equal(upstream?.edge.source, "a");
  assert.deepEqual(upstream?.via, ["r1", "r2"]);
  edges = removeReroutesFromEdges(edges, new Set(["r1"]));
  assert.deepEqual(edges[0].data, { via: ["r2"] });
  assert.equal(edges.length, 1);
  assert.deepEqual([edges[0].source, edges[0].target], ["a", "b"]);
});

test("reroutes round-trip through save and import, and the backend payload is unchanged", () => {
  for (const { name, payload } of learningRecipes) {
    const snapshot = importSnapshot(payload);
    const build = (edges: typeof snapshot.edges, reroutes = {}) =>
      buildRecipePayload(
        snapshot.configs,
        snapshot.nodes,
        edges,
        snapshot.processors,
        snapshot.layoutDirection,
        snapshot.auxNodePositions,
        reroutes,
      ).payload;
    const plain = build(snapshot.edges);
    const target = snapshot.edges[0];
    if (!target) {
      continue;
    }
    const reroutes = { r90: { x: 12, y: 34 }, r91: { x: 56, y: 78 } };
    const routed = build(
      insertRerouteIntoEdge(insertRerouteIntoEdge(snapshot.edges, target.id, "r90"), `${target.id}~1`, "r91"),
      reroutes,
    );
    assert.deepEqual(routed.recipe, plain.recipe, `${name}: the recipe itself is untouched`);
    assert.deepEqual(routed.ui.reroutes, [
      { id: "r90", x: 12, y: 34 },
      { id: "r91", x: 56, y: 78 },
    ]);
    assert.deepEqual(withoutReroutes(routed), plain, `${name}: runs carry no reroutes`);

    const reopened = importSnapshot(routed);
    assert.deepEqual(reopened.reroutes, reroutes, name);
    const reopenedEdge = reopened.edges.find(
      (e) => e.source === target.source && e.target === target.target,
    );
    assert.deepEqual(reopenedEdge?.data, { via: ["r90", "r91"] }, name);
  }
});

test("wires around the focused card stand out and the rest dim", () => {
  assert.equal(wireEmphasis(new Set(), ["a", "b"]), "none");
  assert.equal(wireEmphasis(new Set(["a"]), ["a", "b"]), "focus");
  assert.equal(wireEmphasis(new Set(["c"]), ["a", "b"]), "dim");
  // A segment between two reroutes still lights up for the cards at either end of its wire.
  assert.equal(wireEmphasis(new Set(["b"]), ["r1", "r2", "a", "b"]), "focus");
});

test("each labelled port gets its own wire end, with no slanted legs", () => {
  const rects = [
    { id: "a", x: 0, y: 0, width: 200, height: 160 },
    { id: "b", x: 420, y: 13.4, width: 200, height: 160 },
  ];
  const wires = [
    { id: "w1", source: "a", target: "b", sourceSide: "right", targetSide: "left", sourceOffset: 40.3, targetOffset: 37.7, net: "a:1" },
    { id: "w2", source: "a", target: "b", sourceSide: "right", targetSide: "left", sourceOffset: 120.1, targetOffset: 61.9, net: "a:2" },
  ] as const;
  const { routes } = routeWires(rects, wires as never);
  for (const wire of wires) {
    const points = routes.get(wire.id)?.points ?? [];
    assert.ok(Math.abs(points[0].y - wire.sourceOffset) <= 0.5, `${wire.id} leaves its port`);
    assert.ok(
      Math.abs(points[points.length - 1].y - (13.4 + wire.targetOffset)) <= 0.5,
      `${wire.id} enters its port`,
    );
    for (let k = 0; k + 1 < points.length; k += 1) {
      const straight =
        Math.abs(points[k].x - points[k + 1].x) < 0.01 ||
        Math.abs(points[k].y - points[k + 1].y) < 0.01;
      assert.ok(straight, `${wire.id} segment ${k} is not slanted: ${JSON.stringify(points)}`);
    }
  }
  const hit = findWireNear(routes.values(), { x: 300, y: routes.get("w1")!.points[1].y + 3 }, 10);
  assert.ok(hit);
});
