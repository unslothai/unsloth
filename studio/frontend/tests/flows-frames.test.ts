// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Frames: a frame inside another draws above it, and cards join or leave a frame by where they sit.

import assert from "node:assert/strict";
import test from "node:test";

import { doc, node } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { cloneFragment, useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { RfDeriver, groupDepths } = await import("../src/features/flows/editor/derive-rf.ts");

const frame = (id: string, rect: [number, number, number, number]) => ({ id, label: id, color: null, rect });

test("an outer frame added after an inner one sits below it", () => {
  const d = doc([], []);
  d.groups = [frame("recipe", [100, 100, 400, 300]), frame("loop", [0, 0, 700, 600])];
  useDocStore.getState().load(d);
  const graph = useDocStore.getState().graph;
  assert.deepEqual([...groupDepths(graph)], [["recipe", 1], ["loop", 0]]);
  const nodes = new RfDeriver().nodes(graph, new Set(), null);
  const z = (id: string) => nodes.find((n) => n.id === `group:${id}`)?.zIndex ?? 0;
  assert.ok(z("recipe") > z("loop"), "inner above outer");
  assert.ok(z("recipe") < 0, "frames stay under the cards");
});

test("side-by-side frames share a level; identical ones stack in document order", () => {
  const d = doc([], []);
  d.groups = [frame("a", [0, 0, 200, 200]), frame("b", [300, 0, 200, 200]), frame("c", [0, 0, 200, 200])];
  useDocStore.getState().load(d);
  assert.deepEqual([...groupDepths(useDocStore.getState().graph)], [["a", 0], ["b", 0], ["c", 1]]);
});

test("a selected frame stays under unselected cards, so their resize handles stay reachable", () => {
  const d = doc([], []);
  d.groups = [frame("outer", [0, 0, 700, 600]), frame("mid", [50, 50, 500, 400]), frame("inner", [100, 100, 300, 200])];
  useDocStore.getState().load(d);
  const nodes = new RfDeriver().nodes(useDocStore.getState().graph, new Set(["group:inner"]), null);
  const z = (id: string) => nodes.find((n) => n.id === `group:${id}`)?.zIndex ?? 0;
  assert.ok(z("outer") < z("mid") && z("mid") < z("inner"), "nesting order kept");
  // React Flow's elevateNodesOnSelect adds SELECTED_NODE_Z (1000); a card sits at 0.
  assert.ok(z("inner") + 1000 < 0);
});

test("a frame's body lets the pointer through to the cards under it", () => {
  const d = doc([], []);
  d.groups = [frame("g", [0, 0, 400, 300])];
  useDocStore.getState().load(d);
  const [g] = new RfDeriver().nodes(useDocStore.getState().graph, new Set(), null);
  assert.equal((g.style as { pointerEvents?: string }).pointerEvents, "none");
  assert.equal((g.style as { width?: number }).width, 400);
});

/** A For Each frame g1 at rect, holding its loop card and any `members`, plus free `others`. */
function loopFrame(
  rect: [number, number, number, number],
  members: ReturnType<typeof node>[] = [],
  others: ReturnType<typeof node>[] = [],
) {
  const d = doc([
    { ...node("fe", "logic.for_each"), group: "g1" },
    ...members.map((m) => ({ ...m, group: "g1" })),
    ...others,
  ]);
  d.groups = [{ id: "g1", label: "Loop", rect }];
  useDocStore.getState().load(d);
  return useDocStore.getState();
}
const groupOf = (id: string) => useDocStore.getState().graph.nodes[id]?.group ?? null;

test("deleting a group frame releases the nodes it held", () => {
  loopFrame([0, 0, 600, 300], [node("t", "text.template", [300, 0])]).removeElements({ groups: ["g1"] });
  assert.equal(groupOf("fe"), null);
  assert.equal(groupOf("t"), null);
});

test("a copied node does not join the original's For Each body", () => {
  const store = loopFrame([0, 0, 600, 300], [node("t", "text.template", [300, 0])]);
  const fragment = cloneFragment(store.graph, ["fe", "t"], [900, 0]);
  assert.equal(fragment.nodes.length, 2);
  for (const copy of fragment.nodes) assert.equal(copy.group, null);
  assert.equal(groupOf("t"), "g1");
});

test("a pasted copy joins the frame it lands in, and only that one", () => {
  const store = loopFrame([0, 0, 800, 400], [node("t", "text.template", [100, 100])]);
  const inside = cloneFragment(store.graph, ["t"], [40, 40]);
  const outside = cloneFragment(store.graph, ["t"], [2000, 0]);
  store.insertFragment(inside);
  useDocStore.getState().insertFragment(outside);
  assert.equal(groupOf(inside.nodes[0].id), "g1");
  assert.equal(groupOf(outside.nodes[0].id), null);
});

test("resizing a frame re-parents the nodes it takes in or leaves out", () => {
  const store = loopFrame(
    [0, 0, 600, 300],
    [node("in", "text.template", [100, 100])],
    [node("out", "text.template", [900, 100])],
  );
  const g = store.graph.groups.g1;
  store.resizeGroup({ ...g, rect: [0, 0, 1300, 300] });
  assert.equal(groupOf("out"), "g1");
  useDocStore.getState().resizeGroup({ ...g, rect: [0, 0, 380, 80] });
  assert.equal(groupOf("fe"), "g1");
  assert.equal(groupOf("in"), null);
  assert.equal(groupOf("out"), null);
});

for (const [id, pos, group] of [
  ["inside", [100, 100], "g1"],
  ["outside", [900, 100], null],
] as const) {
  test(`a node added ${id} a frame ${group ? "joins" : "stays out of"} it`, () => {
    loopFrame([0, 0, 600, 300]).addNode(node(id, "text.template", [...pos]));
    assert.equal(groupOf(id), group);
  });
}
