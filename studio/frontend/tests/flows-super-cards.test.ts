// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { catalog, doc, edge, node, specs } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore, documentFromState, documentSignature } = await import(
  "../src/features/flows/stores/doc-store.ts"
);
const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");
const { edgeHandles } = await import(
  "../src/features/flows/editor/connection-rules.ts"
);
const {
  applySuperCards,
  collapsedGroups,
  foldLayout,
  rollUpStatus,
  superCardTitle,
  withSuperCard,
} =
  await import("../src/features/flows/editor/super-cards.ts");

// prompt -> [template -> image] -> preview, with a second preview fed by the template.
function comic() {
  const d = doc(
    [
      node("prompt", "input.text", [0, 0]),
      { ...node("tpl", "text.template", [300, 40]), group: "g", label: "Panel script" },
      { ...node("img", "image.generate", [600, 0]), group: "g" },
      node("view", "output.preview", [900, 0]),
      node("raw", "output.preview", [900, 300]),
    ],
    [
      edge("e1", "prompt", "text", "tpl", "values"),
      edge("e2", "tpl", "text", "img", "prompt"),
      edge("e3", "img", "image", "view", "value"),
      edge("e4", "tpl", "text", "raw", "value"),
    ],
  );
  d.groups = [{ id: "g", label: "Comic panel", color: "#8b5cf6", rect: [280, -40, 640, 320] }];
  return d;
}

function view(collapsed: string[]) {
  const state = useDocStore.getState();
  const deriver = new RfDeriver();
  const nodes = deriver.nodes(state.graph, new Set(), null);
  const edges = deriver.edges(state.graph, new Set(), () => "any");
  return applySuperCards(
    nodes,
    edges,
    state.graph,
    collapsedGroups({ super_cards: collapsed }, state.graph),
    specs,
    catalog.port_colors,
  );
}

test("a collapsed group shows as one card with the wires that cross its frame as ports", () => {
  useDocStore.getState().load(comic());
  const { nodes, edges } = view(["g"]);
  assert.deepEqual(
    nodes.map((n) => n.id).sort(),
    ["group:g", "prompt", "raw", "view"],
  );
  const card = nodes.find((n) => n.id === "group:g");
  assert.equal(card?.type, "flowSuperCard");
  assert.deepEqual(card?.position, { x: 280, y: -40 });
  const data = card?.data as unknown as {
    memberIds: string[];
    inputs: { handleId: string; label: string }[];
    outputs: { handleId: string; label: string }[];
  };
  assert.deepEqual(data.memberIds, ["tpl", "img"]);
  assert.deepEqual(
    data.inputs.map((p) => p.label),
    ["Panel script · Values"],
  );
  // Sorted by where the inner block sits, top first.
  assert.deepEqual(
    data.outputs.map((p) => p.label),
    ["Generate Image · Image", "Panel script · Text"],
  );
  // The inner wire is gone; the three crossing wires end on the card's own handles.
  assert.deepEqual(edges.map((e) => e.id).sort(), ["e1", "e3", "e4"]);
  for (const e of edges) {
    const handles = [
      ...(e.source === "group:g" ? [e.sourceHandle] : []),
      ...(e.target === "group:g" ? [e.targetHandle] : []),
    ];
    assert.equal(handles.length, 1);
    const ports = [...data.inputs, ...data.outputs].map((p) => p.handleId);
    assert.ok(ports.includes(handles[0] as string), `${e.id} lands on a card port`);
  }
});

test("wires from one inner port to several outside blocks share one output port", () => {
  const d = comic();
  d.edges.push(edge("e5", "img", "image", "raw", "value"));
  useDocStore.getState().load(d);
  const card = view(["g"]).nodes.find((n) => n.id === "group:g");
  const outputs = (card?.data as unknown as { outputs: { label: string }[] }).outputs;
  assert.equal(outputs.filter((p) => p.label === "Generate Image · Image").length, 1);
});

test("an open group renders exactly as before", () => {
  useDocStore.getState().load(comic());
  const state = useDocStore.getState();
  const deriver = new RfDeriver();
  const nodes = deriver.nodes(state.graph, new Set(), null);
  const edges = deriver.edges(state.graph, new Set(), () => "any");
  const out = applySuperCards(nodes, edges, state.graph, new Set(), specs);
  assert.equal(out.nodes, nodes);
  assert.equal(out.edges, edges);
  // A stale id for a deleted group is ignored.
  assert.equal(collapsedGroups({ super_cards: ["gone"] }, state.graph).size, 0);
});

test("collapse and expand keep every node, edge and position; only editor state changes", () => {
  useDocStore.getState().load(comic());
  const before = documentFromState(useDocStore.getState());
  const store = useDocStore.getState();
  store.setMeta({ editor: withSuperCard(store.meta.editor, "g", true) });
  const collapsed = documentFromState(useDocStore.getState());
  assert.deepEqual(collapsed.editor, { super_cards: ["g"] });
  const { editor: _a, ...collapsedRest } = collapsed;
  const { editor: _b, ...beforeRest } = before;
  assert.deepEqual(collapsedRest, beforeRest);
  // Autosave sees the collapse as an edit.
  assert.notEqual(documentSignature(collapsed), documentSignature(before));

  const again = useDocStore.getState();
  again.setMeta({ editor: withSuperCard(again.meta.editor, "g", false) });
  assert.deepEqual(documentFromState(useDocStore.getState()), before);
  const flat = view([]);
  assert.deepEqual(
    flat.edges.map((e) => [e.id, e.source, e.sourceHandle, e.target, e.targetHandle]),
    before.edges.map((e) => {
      const h = edgeHandles(e);
      return [e.id, e.source, h.sourceHandle, e.target, h.targetHandle];
    }),
  );
});

test("undo puts a collapsed card back open, redo folds it again", () => {
  useDocStore.getState().load(comic());
  const store = useDocStore.getState();
  store.setMeta({ editor: withSuperCard(store.meta.editor, "g", true) });
  assert.ok(collapsedGroups(useDocStore.getState().meta.editor, useDocStore.getState().graph).has("g"));
  useDocStore.getState().undo();
  assert.equal(
    collapsedGroups(useDocStore.getState().meta.editor, useDocStore.getState().graph).size,
    0,
  );
  useDocStore.getState().redo();
  assert.ok(collapsedGroups(useDocStore.getState().meta.editor, useDocStore.getState().graph).has("g"));
});

test("two collapsed cards wire card to card", () => {
  const d = comic();
  d.nodes = d.nodes.map((n) => (n.id === "view" ? { ...n, group: "h" } : n));
  d.groups.push({ id: "h", label: "Out", rect: [880, -40, 300, 200] });
  useDocStore.getState().load(d);
  const { edges } = view(["g", "h"]);
  const e3 = edges.find((e) => e.id === "e3");
  assert.equal(e3?.source, "group:g");
  assert.equal(e3?.target, "group:h");
});

test("the card rolls its blocks' run status into one", () => {
  assert.equal(rollUpStatus([undefined, undefined]).status, "idle");
  assert.deepEqual(rollUpStatus(["done", "running"]), { status: "running", done: 1 });
  assert.equal(rollUpStatus(["done", "queued"]).status, "running");
  assert.equal(rollUpStatus(["queued", "idle"]).status, "queued");
  assert.equal(rollUpStatus(["done", "failed"]).status, "failed");
  assert.equal(rollUpStatus(["done", "cached"]).status, "done");
  assert.equal(rollUpStatus(["skipped"]).status, "idle");
  // A recipe's columns stay idle while its Recipe Dataset step runs and finishes.
  assert.equal(rollUpStatus([undefined, "idle", "done"]).status, "done");
  assert.equal(rollUpStatus(["done", "cancelled"]).status, "cancelled");
});

test("a recipe group's card is titled as a recipe", () => {
  assert.equal(superCardTitle({ label: "Text to SQL", kind: "recipe" }), "Recipe: Text to SQL");
  assert.equal(superCardTitle({ label: "Comic panel" }), "Comic panel");
});

test("auto layout places a collapsed card as one block and moves its members with it", () => {
  useDocStore.getState().load(comic());
  const store = useDocStore.getState();
  store.setMeta({ editor: withSuperCard(store.meta.editor, "g", true) });
  const state = useDocStore.getState();
  const request = {
    direction: "LR" as const,
    nodes: state.graph.nodeOrder.map((id) => ({ id, width: 260, height: 140 })),
    edges: state.graph.edgeOrder.map((id) => ({
      source: state.graph.edges[id].source,
      target: state.graph.edges[id].target,
    })),
  };
  const folded = foldLayout(request, state, () => ({ width: 288, height: 120 }));
  assert.deepEqual(
    folded.request.nodes.map((n) => n.id).sort(),
    ["group:g", "prompt", "raw", "view"],
  );
  assert.deepEqual(
    folded.request.edges.map((e) => `${e.source}>${e.target}`).sort(),
    ["group:g>raw", "group:g>view", "prompt>group:g"],
  );
  const result = folded.unfold({ "group:g": [380, 60], prompt: [0, 0] });
  assert.deepEqual(result["group:g"], [380, 60]);
  assert.deepEqual(result.tpl, [400, 140]);
  assert.deepEqual(result.img, [700, 100]);
});

test("other things drawn inside a collapsed frame fold with it", () => {
  useDocStore.getState().load(comic());
  const state = useDocStore.getState();
  const deriver = new RfDeriver();
  const nodes = [
    ...deriver.nodes(state.graph, new Set(), null),
    { id: "rr1", type: "flowReroute", position: { x: 500, y: 200 }, data: { nodeId: "rr1" } },
    { id: "note:n1", type: "flowNote", position: { x: 300, y: 200 }, data: { noteId: "n1" } },
    { id: "note:n2", type: "flowNote", position: { x: 300, y: 900 }, data: { noteId: "n2" } },
  ];
  const edges = [
    ...deriver.edges(state.graph, new Set(), () => "any"),
    { id: "w1", source: "prompt", target: "rr1", sourceHandle: "o:text", targetHandle: "in" },
  ];
  const out = applySuperCards(nodes, edges, state.graph, new Set(["g"]), specs);
  assert.ok(!out.nodes.some((n) => n.id === "rr1" || n.id === "note:n1"));
  assert.ok(out.nodes.some((n) => n.id === "note:n2"));
  const w1 = out.edges.find((e) => e.id === "w1");
  assert.equal(w1?.target, "group:g");
});
