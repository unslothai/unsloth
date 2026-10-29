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
  foldLayout,
  multiSelectKeys,
  readSuperCards,
  rollUpStatus,
  superCardDragMembers,
  superCardTitle,
} = await import("../src/features/flows/editor/super-cards.ts");
const { combineSelection, setSuperCardOpen, ungroupSuperCard } = await import(
  "../src/features/flows/editor/super-card-actions.ts"
);
const { useUiStore } = await import("../src/features/flows/stores/ui-store.ts");

// prompt -> [template -> image] -> preview, with a second preview fed by the template.
function comic() {
  const d = doc(
    [
      node("prompt", "input.text", [0, 0]),
      {
        ...node("tpl", "text.template", [300, 40]),
        group: "g",
        label: "Panel script",
      },
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
  d.groups = [
    {
      id: "g",
      label: "Comic panel",
      color: "#8b5cf6",
      rect: [280, -40, 640, 320],
    },
  ];
  return d;
}

function view(
  extra: {
    id: string;
    type: string;
    position: { x: number; y: number };
    data: Record<string, unknown>;
  }[] = [],
) {
  const state = useDocStore.getState();
  const deriver = new RfDeriver();
  const nodes = [...deriver.nodes(state.graph, new Set(), null), ...extra];
  const edges = deriver.edges(state.graph, new Set(), () => "any");
  return applySuperCards(
    nodes as never,
    edges,
    state.graph,
    readSuperCards(state.meta.editor, state.graph),
    specs,
    catalog.port_colors,
  );
}

function select(...ids: string[]) {
  useUiStore.getState().setSelection(ids, []);
}

function cards() {
  const { meta, graph } = useDocStore.getState();
  return readSuperCards(meta.editor, graph);
}

function withoutEditor() {
  const { editor: _e, ...rest } = documentFromState(useDocStore.getState());
  return rest;
}

test("Cmd adds to the selection on a Mac, Ctrl on Windows and Linux, Shift everywhere", () => {
  assert.deepEqual(multiSelectKeys("MacIntel"), ["Meta", "Shift"]);
  assert.deepEqual(multiSelectKeys("iPad"), ["Meta", "Shift"]);
  assert.deepEqual(multiSelectKeys("Win32"), ["Control", "Shift"]);
  assert.deepEqual(multiSelectKeys("Linux x86_64"), ["Control", "Shift"]);
  assert.deepEqual(multiSelectKeys(""), ["Control", "Shift"]);
});

test("combining two or more selected blocks makes one closed card and selects it", () => {
  useDocStore.getState().load(comic());
  const before = withoutEditor();
  select("tpl");
  assert.equal(combineSelection(), null, "one block is not a card");
  select("tpl", "img");
  assert.equal(combineSelection(), "sc1");
  // Exactly a group's blocks: the card takes the group's name.
  assert.deepEqual(cards(), [
    { id: "sc1", title: "Comic panel", members: ["tpl", "img"] },
  ]);
  assert.deepEqual(useUiStore.getState().selectedNodes, ["super:sc1"]);
  // Only editor state changed, so the compiled plan is the same.
  assert.deepEqual(withoutEditor(), before);
  // A block already in a card can't go into a second one.
  select("img", "view");
  assert.equal(combineSelection(), null);
  select("prompt", "raw");
  assert.equal(combineSelection(), "sc2");
  assert.equal(cards()[1].title, "Super card 2");
});

test("a closed card shows the wires that cross into it as ports and hides its group frame", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  const { nodes, edges } = view();
  assert.deepEqual(nodes.map((n) => n.id).sort(), [
    "prompt",
    "raw",
    "super:sc1",
    "view",
  ]);
  const card = nodes.find((n) => n.id === "super:sc1");
  assert.equal(card?.type, "flowSuperCard");
  assert.deepEqual(card?.position, { x: 300, y: 0 });
  const data = card?.data as unknown as {
    memberIds: string[];
    folded: { id: string; x: number; y: number }[];
    inputs: { handleId: string; label: string }[];
    outputs: { handleId: string; label: string }[];
  };
  assert.deepEqual(data.memberIds, ["tpl", "img"]);
  assert.deepEqual(data.folded, [{ id: "group:g", x: 280, y: -40 }]);
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
  const ports = [...data.inputs, ...data.outputs].map((p) => p.handleId);
  for (const e of edges) {
    const handle = e.source === "super:sc1" ? e.sourceHandle : e.targetHandle;
    assert.ok(ports.includes(handle as string), `${e.id} lands on a card port`);
  }
});

test("wires from one inner port to several outside blocks share one output port", () => {
  const d = comic();
  d.edges.push(edge("e5", "img", "image", "raw", "value"));
  useDocStore.getState().load(d);
  select("tpl", "img");
  combineSelection();
  const card = view().nodes.find((n) => n.id === "super:sc1");
  const outputs = (card?.data as unknown as { outputs: { label: string }[] })
    .outputs;
  assert.equal(
    outputs.filter((p) => p.label === "Generate Image · Image").length,
    1,
  );
});

test("opening a card renders exactly as before; collapse folds it again; ungroup drops it", () => {
  useDocStore.getState().load(comic());
  const flat = view();
  select("tpl", "img");
  combineSelection();
  setSuperCardOpen("sc1", true);
  assert.equal(cards()[0].open, true);
  assert.deepEqual(useUiStore.getState().selectedNodes, ["tpl", "img"]);
  const open = view();
  assert.deepEqual(
    open.nodes.map((n) => n.id),
    flat.nodes.map((n) => n.id),
  );
  assert.deepEqual(
    open.edges.map((e) => [
      e.id,
      e.source,
      e.sourceHandle,
      e.target,
      e.targetHandle,
    ]),
    useDocStore.getState().graph.edgeOrder.map((id) => {
      const e = useDocStore.getState().graph.edges[id];
      const h = edgeHandles(e);
      return [e.id, e.source, h.sourceHandle, e.target, h.targetHandle];
    }),
  );
  setSuperCardOpen("sc1", false);
  assert.ok(view().nodes.some((n) => n.id === "super:sc1"));
  ungroupSuperCard("sc1");
  assert.deepEqual(cards(), []);
  assert.equal(useDocStore.getState().meta.editor?.super_cards, undefined);
});

test("combine, open and ungroup are edits: autosave sees them and undo/redo walk them back", () => {
  useDocStore.getState().load(comic());
  const plain = documentSignature(documentFromState(useDocStore.getState()));
  select("tpl", "img");
  combineSelection();
  assert.notEqual(
    documentSignature(documentFromState(useDocStore.getState())),
    plain,
  );
  setSuperCardOpen("sc1", true);
  useDocStore.getState().undo();
  assert.equal(cards()[0].open, undefined);
  useDocStore.getState().undo();
  assert.deepEqual(cards(), []);
  useDocStore.getState().redo();
  assert.deepEqual(
    cards().map((c) => c.id),
    ["sc1"],
  );
  useDocStore.getState().redo();
  assert.equal(cards()[0].open, true);
});

test("flows saved with the earlier group-id format still load as cards", () => {
  const d = comic();
  d.editor = { super_cards: ["g", "gone"] };
  useDocStore.getState().load(d);
  assert.deepEqual(cards(), [
    { id: "g", title: "Comic panel", members: ["tpl", "img"] },
  ]);
  assert.ok(view().nodes.some((n) => n.id === "super:g"));
});

test("two closed cards wire card to card", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  select("view", "raw");
  combineSelection();
  const e3 = view().edges.find((e) => e.id === "e3");
  assert.equal(e3?.source, "super:sc1");
  assert.equal(e3?.target, "super:sc2");
});

test("dragging a card carries its blocks and its folded group frame", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  const card = view().nodes.find((n) => n.id === "super:sc1");
  const members = superCardDragMembers(
    card as never,
    useDocStore.getState().graph,
  );
  assert.deepEqual(Object.fromEntries(members ?? []), {
    tpl: { x: 300, y: 40 },
    img: { x: 600, y: 0 },
    "group:g": { x: 280, y: -40 },
  });
});

test("the card rolls its blocks' run status into one", () => {
  assert.equal(rollUpStatus([undefined, undefined]).status, "idle");
  assert.deepEqual(rollUpStatus(["done", "running", undefined]), {
    status: "running",
    done: 1,
    total: 2,
  });
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
  assert.equal(
    superCardTitle({ label: "Text to SQL", kind: "recipe" }),
    "Recipe: Text to SQL",
  );
  assert.equal(superCardTitle({ label: "Comic panel" }), "Comic panel");
});

test("auto layout places a closed card as one block and moves its blocks with it", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  const state = useDocStore.getState();
  const request = {
    direction: "LR" as const,
    nodes: state.graph.nodeOrder.map((id) => ({ id, width: 260, height: 140 })),
    edges: state.graph.edgeOrder.map((id) => ({
      source: state.graph.edges[id].source,
      target: state.graph.edges[id].target,
    })),
  };
  const folded = foldLayout(request, state, () => ({
    width: 288,
    height: 120,
  }));
  assert.deepEqual(folded.request.nodes.map((n) => n.id).sort(), [
    "prompt",
    "raw",
    "super:sc1",
    "view",
  ]);
  assert.deepEqual(
    folded.request.edges.map((e) => `${e.source}>${e.target}`).sort(),
    ["prompt>super:sc1", "super:sc1>raw", "super:sc1>view"],
  );
  const result = folded.unfold({ "super:sc1": [380, 60], prompt: [0, 0] });
  assert.equal(result["super:sc1"], undefined);
  assert.deepEqual(result.tpl, [380, 100]);
  assert.deepEqual(result.img, [680, 60]);
});

test("notes and reroutes drawn among a card's blocks fold with it", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  const state = useDocStore.getState();
  const deriver = new RfDeriver();
  const nodes = [
    ...deriver.nodes(state.graph, new Set(), null),
    {
      id: "rr1",
      type: "flowReroute",
      position: { x: 500, y: 100 },
      data: { nodeId: "rr1" },
    },
    {
      id: "note:n1",
      type: "flowNote",
      position: { x: 320, y: 100 },
      data: { noteId: "n1" },
    },
    {
      id: "note:n3",
      type: "flowNote",
      position: { x: 700, y: 220 },
      data: { noteId: "n3" },
    },
    {
      id: "note:n2",
      type: "flowNote",
      position: { x: 300, y: 900 },
      data: { noteId: "n2" },
    },
  ];
  const edges = [
    ...deriver.edges(state.graph, new Set(), () => "any"),
    {
      id: "w1",
      source: "prompt",
      target: "rr1",
      sourceHandle: "o:text",
      targetHandle: "in",
    },
  ];
  const out = applySuperCards(
    nodes as never,
    edges,
    state.graph,
    readSuperCards(state.meta.editor, state.graph),
    specs,
  );
  assert.ok(!out.nodes.some((n) => n.id === "rr1" || n.id === "note:n1"));
  // Inside the folded group frame, below the blocks.
  assert.ok(!out.nodes.some((n) => n.id === "note:n3"));
  assert.ok(out.nodes.some((n) => n.id === "note:n2"));
  // Dragging the card carries them, where they sit.
  const card = out.nodes.find((n) => n.id === "super:sc1");
  const members = superCardDragMembers(card as never, state.graph);
  assert.deepEqual(members?.get("rr1"), { x: 500, y: 100 });
  assert.deepEqual(members?.get("note:n1"), { x: 320, y: 100 });
  assert.equal(out.edges.find((e) => e.id === "w1")?.target, "super:sc1");
});
