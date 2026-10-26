// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, edge, node } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore, cloneFragment } = await import(
  "../src/features/flows/stores/doc-store.ts"
);

function fresh() {
  useDocStore
    .getState()
    .load(
      doc(
        [
          node("a", "input.text"),
          node("b", "llm.generate", [300, 0]),
          node("c", "output.preview", [600, 0]),
        ],
        [
          edge("e1", "a", "text", "b", "prompt"),
          edge("e2", "b", "text", "c", "value"),
        ],
      ),
    );
  return useDocStore.getState();
}

test("undo and redo walk the history", () => {
  const s = fresh();
  s.addNode(node("d", "input.number"));
  s.removeElements({ nodes: ["a"] });
  assert.equal(useDocStore.getState().graph.nodes.a, undefined);
  assert.equal(
    useDocStore.getState().graph.edges.e1,
    undefined,
    "removing a node drops its edges",
  );
  useDocStore.getState().undo();
  assert.ok(useDocStore.getState().graph.nodes.a);
  assert.ok(useDocStore.getState().graph.edges.e1);
  useDocStore.getState().undo();
  assert.equal(useDocStore.getState().graph.nodes.d, undefined);
  useDocStore.getState().redo();
  assert.ok(useDocStore.getState().graph.nodes.d);
  assert.equal(useDocStore.getState().future.length, 1);
  useDocStore.getState().addNode(node("e", "input.number"));
  assert.equal(
    useDocStore.getState().future.length,
    0,
    "a new edit clears redo",
  );
});

test("typing into one field is a single undo step", () => {
  const s = fresh();
  for (const text of ["h", "he", "hel", "hello"])
    s.updateParams("b", { prompt: text }, "param:b:prompt");
  assert.equal(useDocStore.getState().past.length, 1);
  useDocStore.getState().updateParams("b", { system: "x" }, "param:b:system");
  assert.equal(
    useDocStore.getState().past.length,
    2,
    "a different field starts a new step",
  );
  useDocStore.getState().undo();
  useDocStore.getState().undo();
  assert.equal(useDocStore.getState().graph.nodes.b.params.prompt, undefined);
});

test("history is capped", () => {
  const s = fresh();
  for (let i = 0; i < 130; i++) s.moveNodes({ a: [i, i] });
  assert.equal(useDocStore.getState().past.length, 100);
});

test("a single-value input keeps one edge; a multi input keeps all", () => {
  const s = fresh();
  s.addNode(node("t2", "input.text"));
  s.connect(edge("e3", "t2", "text", "b", "prompt"), true);
  const edges = Object.values(useDocStore.getState().graph.edges).filter(
    (e) => e.target === "b" && e.target_port === "prompt",
  );
  assert.deepEqual(
    edges.map((e) => e.id),
    ["e3"],
  );
  useDocStore.getState().connect(edge("e4", "a", "text", "c", "value"), false);
  const multi = Object.values(useDocStore.getState().graph.edges).filter(
    (e) => e.target === "c",
  );
  assert.equal(multi.length, 2);
  useDocStore.getState().connect(edge("e5", "a", "text", "c", "value"), false);
  assert.equal(
    useDocStore.getState().graph.edges.e5,
    undefined,
    "an exact duplicate is ignored",
  );
});

test("each story path leaves a handle once", () => {
  const s = fresh();
  s.addNode(node("s1", "story.scene"));
  s.addNode(node("s2", "story.scene"));
  s.addNode(node("s3", "story.scene"));
  s.connect(edge("f1", "s1", "next", "s2", "flow", "flow"), false);
  useDocStore
    .getState()
    .connect(edge("f2", "s1", "next", "s3", "flow", "flow"), false);
  const out = Object.values(useDocStore.getState().graph.edges).filter(
    (e) => e.source === "s1",
  );
  assert.deepEqual(
    out.map((e) => e.target),
    ["s3"],
  );
});

test("frames move in the same undo step as nodes", () => {
  const s = fresh();
  s.upsertGroup({ id: "g", label: "G", rect: [0, 0, 100, 100] });
  const steps = useDocStore.getState().past.length;
  useDocStore
    .getState()
    .moveNodes({ "group:g": [50, 60], a: [55, 65] }, { a: "g" });
  const g = useDocStore.getState().graph;
  assert.deepEqual(g.groups.g.rect, [50, 60, 100, 100]);
  assert.equal(g.nodes.a.group, "g");
  assert.equal(useDocStore.getState().past.length, steps + 1);
});

test("duplicating keeps internal edges with fresh ids", () => {
  const s = fresh();
  const fragment = cloneFragment(s.graph, ["a", "b"], [40, 40]);
  assert.equal(fragment.nodes.length, 2);
  assert.equal(fragment.edges.length, 1);
  const ids = new Set(fragment.nodes.map((n) => n.id));
  assert.ok(!ids.has("a") && !ids.has("b"));
  assert.ok(
    ids.has(fragment.edges[0].source) && ids.has(fragment.edges[0].target),
  );
  assert.deepEqual(fragment.nodes[0].pos, [40, 40]);
  useDocStore.getState().insertFragment(fragment);
  assert.equal(useDocStore.getState().graph.nodeOrder.length, 5);
});
