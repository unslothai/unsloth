// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
  boxCovers,
  foldLayout,
  multiSelectKeys,
  readSuperCards,
  rollUpStatus,
  selectedBlocks,
  superCardDragMembers,
  superCardTitle,
} = await import("../src/features/flows/editor/super-cards.ts");
const { combineSelection, setSuperCardOpen, ungroupSuperCard } = await import(
  "../src/features/flows/editor/super-card-actions.ts"
);
const { useUiStore } = await import("../src/features/flows/stores/ui-store.ts");
const { rememberSizes, sizeLookup } = await import(
  "../src/features/flows/editor/layout.ts"
);

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

/** Cards as saved before they recorded their notes and reroutes: these fold by position. */
function legacyCards() {
  const { meta } = useDocStore.getState();
  const raw = (meta.editor?.super_cards as Record<string, unknown>[]) ?? [];
  useDocStore.getState().setMeta({
    editor: { ...meta.editor, super_cards: raw.map(({ extras: _e, ...rest }) => rest) },
  });
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
    { id: "sc1", title: "Comic panel", members: ["tpl", "img"], extras: [] },
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

test("a closed card selected with more blocks is absorbed: cards stay one level deep", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  select("super:sc1");
  assert.equal(combineSelection(), null, "a card alone is not a new card");
  select("super:sc1", "view");
  assert.equal(combineSelection(), "sc2");
  assert.deepEqual(cards(), [
    { id: "sc2", title: "Comic panel", members: ["tpl", "img", "view"], extras: [] },
  ]);
});

test("a selection box drawn inside a group leaves the group; one around it takes it", () => {
  const group = { x: 100, y: 100, width: 400, height: 300 };
  // Pane at translate(50, 20), zoom 0.5: flow 100 is screen 100.
  assert.equal(boxCovers({ x: 90, y: 60, width: 220, height: 170 }, [50, 20, 0.5], group), true);
  assert.equal(boxCovers({ x: 120, y: 90, width: 100, height: 100 }, [50, 20, 0.5], group), false);
});

test("Cmd/Ctrl drag draws a selection box, and a closed card deletes with its blocks", () => {
  const src = readFileSync(new URL("../src/features/flows/editor/canvas.tsx", import.meta.url), "utf8");
  assert.match(src, /selectionKeyCode=\{MULTI_SELECT_KEYS\}/);
  // A frame is settled against the final box, since React Flow won't offer it again.
  assert.match(src, /onSelectionEnd=\{onSelectionEnd\}/);
  assert.match(src, /if \(boxHoldsGroup\(state, id, pressed\)\) next\.add\(id\);/);
  assert.match(src, /data\.memberIds, \.\.\.data\.folded/);
  const lib = readFileSync(new URL("../src/features/flows/editor/super-cards.ts", import.meta.url), "utf8");
  const cardNode = lib.slice(lib.indexOf('type: "flowSuperCard"'), lib.indexOf('dragHandle: ".flow-node-drag"'));
  assert.doesNotMatch(cardNode, /deletable: false/);
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
  // The folded group's colour carries onto the card.
  assert.equal((card?.data as { color?: string }).color, "#8b5cf6");
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
  // Every selected card moves with the drag, not only the one under the pointer.
  const canvas = readFileSync(new URL("../src/features/flows/editor/canvas.tsx", import.meta.url), "utf8");
  assert.match(canvas, /for \(const n of dragged\.length \? dragged : \[node\]\) \{\s*const folded = superCardDragMembers\(n, doc\);/);
  assert.match(canvas, /const group = dragMembers\.current\.get\(change\.id\);/);
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
  const d = comic();
  d.notes = [
    { id: "n1", text: "inside", rect: [320, 100, 100, 60] },
    { id: "n2", text: "outside", rect: [0, 900, 100, 60] },
  ];
  useDocStore.getState().load(d);
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
  // The folded group frame and the note among the blocks move by the same amount.
  assert.deepEqual(result["group:g"], [360, 20]);
  assert.deepEqual(result["note:n1"], [400, 160]);
  assert.equal(result["note:n2"], undefined);
});

test("tidy up folds a group the card holds whole, so the card keeps its wires", async () => {
  const { tidyRequest } = await import("../src/features/flows/editor/tidy.ts");
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  const state = useDocStore.getState();
  const request = tidyRequest({
    graph: state.graph,
    size: () => ({ width: 260, height: 140 }),
    portY: () => undefined,
    wireStyle: "curved",
  });
  assert.ok(request.nodes.some((n) => n.id === "group:g"));
  const folded = foldLayout(request, state, () => undefined);
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
});

test("auto layout moves a note a wide block folded, after the block left React Flow", () => {
  const d = doc(
    [node("a", "input.text", [0, 0]), node("b", "image.generate", [300, 0])],
    [],
  );
  // Centred inside b's measured 500x400 box, outside the 260x140 default.
  d.notes = [{ id: "n1", text: "beside", rect: [700, 250, 100, 60] }];
  useDocStore.getState().load(d);
  // The mounted canvas has measured b by the time the card closes.
  rememberSizes(new Map([["b", { width: 500, height: 400 }]]));
  select("a", "b");
  combineSelection();
  const state = useDocStore.getState();
  const request = {
    direction: "LR" as const,
    nodes: state.graph.nodeOrder.map((id) => ({ id, width: 260, height: 140 })),
    edges: [],
  };
  // A closed card's blocks are not in React Flow, so it reports no size for them.
  const lookup = sizeLookup(() => undefined);
  const result = foldLayout(request, state, lookup).unfold({
    "super:sc1": [100, 50],
  });
  assert.deepEqual(result.b, [400, 50]);
  assert.deepEqual(result["note:n1"], [800, 300]);
  // The sizes belong to the mounted canvas: another flow's canvas, or none, brings its own.
  rememberSizes(new Map([["a", { width: 260, height: 140 }]]));
  assert.equal(lookup("b"), undefined);
  rememberSizes(new Map());
  assert.equal(lookup("a"), undefined);
});

test("notes and reroutes drawn among a card's blocks fold with it", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  legacyCards();
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

test("a note in one card's folded frame stays with that card where another card's blocks overlap it", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  select("view", "raw");
  combineSelection();
  legacyCards();
  const out = view([
    {
      id: "note:edge",
      type: "flowNote",
      position: { x: 910, y: 100 },
      data: { noteId: "edge" },
    },
  ]);
  const folded = (id: string) =>
    (
      out.nodes.find((n) => n.id === id)?.data as unknown as {
        folded: { id: string }[];
      }
    ).folded.map((f) => f.id);
  assert.ok(folded("super:sc1").includes("note:edge"));
  assert.ok(!folded("super:sc2").includes("note:edge"));
});

test("a Mac's Ctrl-click (a right-click to the browser) adds or drops a card from the selection", async () => {
  const { toggledSelection, isApplePlatform } = await import("../src/features/flows/editor/super-cards.ts");
  assert.deepEqual(toggledSelection(["a"], "b"), ["a", "b"]);
  assert.deepEqual(toggledSelection(["a", "b"], "a"), ["b"]);
  assert.deepEqual(toggledSelection([], "a"), ["a"]);
  assert.equal(isApplePlatform("MacIntel"), true);
  assert.equal(isApplePlatform("Win32"), false);
  const canvas = readFileSync(new URL("../src/features/flows/editor/canvas.tsx", import.meta.url), "utf8");
  assert.match(canvas, /onNodeContextMenu=\{\(event, node\) => \{\s*if \(!APPLE \|\| !event\.ctrlKey\) return;\s*event\.preventDefault\(\);/);
});

test("copy and duplicate take the blocks inside a selected closed card", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  const { graph, meta } = useDocStore.getState();
  assert.deepEqual(
    selectedBlocks(["super:sc1", "view", "gone"], meta.editor, graph),
    ["tpl", "img", "view"],
  );
  // Cut removes what copy took, not the card's own selection id.
  const shortcuts = readFileSync(new URL("../src/features/flows/editor/use-editor-shortcuts.ts", import.meta.url), "utf8");
  assert.match(shortcuts, /const fragment = copySelection\(\);\s*if \(fragment\) \{[^}]*removeNodes\(fragment\.nodes\.map\(\(n\) => n\.id\)\)/);
});

test("a block added where a folded frame sits stays out of it", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  const doc0 = useDocStore.getState();
  doc0.addNode({ ...node("new", "output.preview", [400, 100]) });
  doc0.insertFragment({ nodes: [node("pasted", "output.preview", [420, 120])], edges: [] });
  const { graph } = useDocStore.getState();
  assert.equal(graph.nodes.new.group ?? null, null);
  assert.equal(graph.nodes.pasted.group ?? null, null);
  // An open frame still takes what lands in it.
  setSuperCardOpen("sc1", true);
  useDocStore.getState().addNode({ ...node("inside", "output.preview", [400, 100]) });
  assert.equal(useDocStore.getState().graph.nodes.inside.group, "g");
  // Dropping a dragged block there skips the folded frame too.
  const canvas = readFileSync(new URL("../src/features/flows/editor/canvas.tsx", import.meta.url), "utf8");
  assert.match(canvas, /if \(folded\.has\(groupId\) && graph\.nodes\[id\]\.group !== groupId\) continue;/);
});

test("tidy up keeps a card's blocks together when the card holds part of a group", async () => {
  const { tidyRequest, tidyPatch } = await import("../src/features/flows/editor/tidy.ts");
  const d = comic();
  d.notes = [{ id: "n1", text: "by the template", rect: [320, 100, 100, 60] }];
  useDocStore.getState().load(d);
  // tpl is in group g with img; raw is outside it.
  select("tpl", "raw");
  combineSelection();
  const state = useDocStore.getState();
  const ctx = {
    graph: state.graph,
    size: () => ({ width: 260, height: 140 }),
    portY: () => undefined,
    wireStyle: "curved" as const,
  };
  const { tidyGraph } = await import("../src/features/flows/editor/super-cards.ts");
  const folded = foldLayout(
    tidyRequest({ ...ctx, graph: tidyGraph(state.graph, state.meta.editor) }),
    state,
    () => undefined,
  );
  // tpl's wires (prompt -> tpl, tpl -> img, tpl -> raw) reach the card, not group g's placeholder.
  const wires = folded.request.edges.map((e) => `${e.source}>${e.target}`).sort();
  assert.deepEqual(wires, ["group:g>view", "prompt>super:sc1", "super:sc1>group:g"]);
  const layout = readFileSync(new URL("../src/features/flows/editor/layout.ts", import.meta.url), "utf8");
  assert.match(layout, /tidyRequest\(\{ \.\.\.ctx, graph: tidyGraph\(ctx\.graph, useDocStore\.getState\(\)\.meta\.editor\) \}\)/);
  const result = folded.unfold({ "super:sc1": [1000, 500], "group:g": [0, 0], prompt: [0, 0], view: [0, 0] });
  const patch = tidyPatch(ctx, result);
  const [tx, ty] = patch.nodes.tpl.pos;
  const [rx, ry] = patch.nodes.raw.pos;
  assert.deepEqual([rx - tx, ry - ty], [900 - 300, 300 - 40]);
  // A note folded into the card moves with the card, not with the nearest loose block.
  const [nx, ny] = patch.notes.n1;
  assert.deepEqual([nx - tx, ny - ty], [320 - 300, 100 - 40]);
});

test("tidy up moves a frame whose blocks all went into different closed cards", async () => {
  const { tidyRequest, tidyPatch } = await import("../src/features/flows/editor/tidy.ts");
  const { tidyGraph } = await import("../src/features/flows/editor/super-cards.ts");
  useDocStore.getState().load(comic());
  // Group g holds tpl and img, and each lands in its own closed card.
  select("tpl", "raw");
  combineSelection();
  select("img", "view");
  combineSelection();
  const state = useDocStore.getState();
  const ctx = {
    graph: state.graph,
    size: () => ({ width: 260, height: 140 }),
    portY: () => undefined,
    wireStyle: "curved" as const,
  };
  const folded = foldLayout(
    tidyRequest({ ...ctx, graph: tidyGraph(state.graph, state.meta.editor) }),
    state,
    () => undefined,
  );
  assert.ok(!folded.request.nodes.some((n) => n.id === "group:g"));
  const ids = cards().map((c) => `super:${c.id}`);
  const patch = tidyPatch(
    ctx,
    folded.unfold({ prompt: [0, 0], [ids[0]]: [2000, 2000], [ids[1]]: [2400, 2000] }),
  );
  // The frame follows its blocks instead of staying where they were.
  const [tx, ty] = patch.nodes.tpl.pos;
  const [gx, gy] = patch.groups.g;
  assert.deepEqual([gx - tx, gy - ty], [280 - 300, -40 - 40]);
});

test("Select All takes closed cards too, so Delete removes them", async () => {
  const { selectAllIds } = await import("../src/features/flows/editor/super-cards.ts");
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  const { meta, graph } = useDocStore.getState();
  const all = selectAllIds(meta.editor, graph);
  assert.ok(all.includes("super:sc1") && all.includes("prompt"));
  const shortcuts = readFileSync(new URL("../src/features/flows/editor/use-editor-shortcuts.ts", import.meta.url), "utf8");
  assert.match(shortcuts, /selectAllIds\(doc\.meta\.editor, doc\.graph\)/);
});

test("a note inside one card's folded frame and another card's blocks moves only with the frame's card", () => {
  const d = comic();
  d.notes = [{ id: "n1", text: "both", rect: [880, 70, 60, 60] }];
  useDocStore.getState().load(d);
  select("tpl", "img");
  combineSelection();
  select("view", "raw");
  combineSelection();
  const state = useDocStore.getState();
  const request = {
    direction: "LR" as const,
    nodes: state.graph.nodeOrder.map((id) => ({ id, width: 260, height: 140 })),
    edges: [],
  };
  // sc1 moves by (100, 0), sc2 by (0, 500).
  const result = foldLayout(request, state, () => undefined).unfold({
    "super:sc1": [400, -0],
    "super:sc2": [900, 500],
  });
  assert.deepEqual(result["note:n1"], [980, 70]);
});

test("a free reroute folded into a card moves with it on Tidy", async () => {
  const { tidyPatch } = await import("../src/features/flows/editor/tidy.ts");
  const d = comic();
  d.editor = { reroutes: [{ id: "rr1", pos: [450, 60] }] };
  useDocStore.getState().load(d);
  select("tpl", "img");
  combineSelection();
  const state = useDocStore.getState();
  const request = {
    direction: "LR" as const,
    nodes: state.graph.nodeOrder.map((id) => ({ id, width: 260, height: 140 })),
    edges: [],
  };
  const result = foldLayout(request, state, () => undefined).unfold({ "super:sc1": [400, 100] });
  assert.deepEqual(result.rr1, [550, 160]);
  const patch = tidyPatch(
    {
      graph: state.graph,
      size: () => ({ width: 260, height: 140 }),
      portY: () => undefined,
      wireStyle: "curved",
    },
    result,
  );
  assert.deepEqual(patch.reroutes.rr1, [550, 160]);
});

test("resizing a visible frame never hands its blocks to a folded one", () => {
  const d = comic();
  d.groups = [...(d.groups ?? []), { id: "outer", label: "Outer", color: "#64748b", rect: [-100, -200, 1500, 900] }];
  d.nodes = [...d.nodes, { ...node("x", "output.preview", [420, 60]), group: "outer" }];
  useDocStore.getState().load(d);
  select("tpl", "img");
  combineSelection();
  useDocStore.getState().resizeGroup({ ...useDocStore.getState().graph.groups.outer, rect: [-100, -200, 1520, 900] });
  const { graph } = useDocStore.getState();
  assert.equal(graph.nodes.x.group, "outer");
  assert.equal(graph.nodes.tpl.group, "g");
});

test("an unmeasured note folds by its saved size, not its corner", () => {
  const d = comic();
  // Its corner is among the card's blocks; its middle is well outside them.
  d.notes = [{ id: "big", text: "big", rect: [320, 100, 800, 600] }];
  useDocStore.getState().load(d);
  select("tpl", "img");
  combineSelection();
  const ids = view().nodes.map((n) => n.id);
  assert.ok(ids.includes("note:big"));
  // Dragging a frame that carries any of a closed card's blocks carries the whole card.
  const canvas = readFileSync(new URL("../src/features/flows/editor/canvas.tsx", import.meta.url), "utf8");
  assert.match(canvas, /if \(card\.open \|\| !card\.members\.some\(\(id\) => members\.has\(id\)\)\) continue;/);
});

test("after a reload a closed card folds by its blocks' saved sizes", () => {
  const d = doc(
    [node("a", "input.text", [0, 0]), { ...node("b", "image.generate", [300, 0]), size: [500, 400] as [number, number] }],
    [],
  );
  d.notes = [{ id: "n1", text: "beside", rect: [700, 250, 100, 60] }];
  useDocStore.getState().load(d);
  select("a", "b");
  combineSelection();
  rememberSizes(new Map());
  assert.ok(!view().nodes.some((n) => n.id === "note:n1"), "the note folds into the card");
  const state = useDocStore.getState();
  const request = {
    direction: "LR" as const,
    nodes: state.graph.nodeOrder.map((id) => ({ id, width: 260, height: 140 })),
    edges: [],
  };
  const result = foldLayout(request, state, () => undefined).unfold({ "super:sc1": [100, 50] });
  assert.deepEqual(result["note:n1"], [800, 300]);
});

test("grouping a selection with a closed card groups the card's blocks", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  const { meta, graph } = useDocStore.getState();
  assert.deepEqual(selectedBlocks(["super:sc1", "view"], meta.editor, graph), ["tpl", "img", "view"]);
  const shortcuts = readFileSync(new URL("../src/features/flows/editor/use-editor-shortcuts.ts", import.meta.url), "utf8");
  const group = shortcuts.slice(shortcuts.indexOf("export function groupSelection"));
  assert.match(group, /const ids = selectedBlocks\(\s*useUiStore\.getState\(\)\.selectedNodes,\s*doc\.meta\.editor,\s*doc\.graph,\s*\);/);
});

test("a rebuilt closed card keeps its measurement, so React Flow does not hide it to measure again", () => {
  useDocStore.getState().load(comic());
  select("tpl", "img");
  combineSelection();
  const state = useDocStore.getState();
  const deriver = new RfDeriver();
  const out = applySuperCards(
    deriver.nodes(state.graph, new Set(), null) as never,
    deriver.edges(state.graph, new Set(), () => "any"),
    state.graph,
    readSuperCards(state.meta.editor, state.graph),
    specs,
    catalog.port_colors,
    new Set(),
    (id) => (id === "super:sc1" ? { width: 300, height: 180 } : undefined),
  );
  assert.deepEqual(out.nodes.find((n) => n.id === "super:sc1")?.measured, { width: 300, height: 180 });
  const canvas = readFileSync(new URL("../src/features/flows/editor/canvas.tsx", import.meta.url), "utf8");
  assert.match(canvas, /useSuperCardView\(nodes, edges, selectedSet, sizes\)/);
});

test("resizing a frame leaves a block hidden in a closed card in its frame, even when the card holds only part of it", () => {
  const d = comic();
  // The card was dragged, so its hidden block's saved spot is now outside the frame.
  d.nodes = d.nodes.map((n) => (n.id === "tpl" ? { ...n, pos: [-600, 500] as [number, number] } : n));
  useDocStore.getState().load(d);
  select("tpl", "prompt");
  combineSelection();
  const g = useDocStore.getState().graph.groups.g;
  useDocStore.getState().resizeGroup({ ...g, rect: [g.rect[0], g.rect[1], g.rect[2] + 10, g.rect[3]] });
  assert.equal(useDocStore.getState().graph.nodes.tpl.group, "g");
  assert.equal(useDocStore.getState().graph.nodes.img.group, "g");
});

test("a note where two folded frames overlap moves with the card the canvas folds it into", () => {
  const d = doc(
    [
      { ...node("a", "input.text", [0, 0]), group: "g1" },
      { ...node("a2", "input.text", [0, 200]), group: "g1" },
      { ...node("b", "input.text", [1000, 0]), group: "g2" },
      { ...node("b2", "input.text", [1000, 200]), group: "g2" },
    ],
    [],
  );
  d.groups = [
    { id: "g1", label: "One", color: "#8b5cf6", rect: [-20, -40, 700, 400] },
    { id: "g2", label: "Two", color: "#8b5cf6", rect: [400, -40, 900, 400] },
  ];
  d.notes = [{ id: "n", text: "both", rect: [500, 100, 100, 50] }];
  useDocStore.getState().load(d);
  // The second frame's card is made first, so card order and block order disagree.
  select("b", "b2");
  combineSelection();
  select("a", "a2");
  combineSelection();
  const [second, first] = cards();
  assert.deepEqual(first.members, ["a", "a2"]);
  assert.deepEqual(second.members, ["b", "b2"]);
  const state = useDocStore.getState();
  const request = {
    direction: "LR" as const,
    nodes: state.graph.nodeOrder.map((id) => ({ id, width: 260, height: 140 })),
    edges: [],
  };
  const result = foldLayout(request, state, () => undefined).unfold({
    [`super:${first.id}`]: [0, 1000],
    [`super:${second.id}`]: [1000, 0],
  });
  // The note was folded into the card made first, and moves with that card, as the canvas shows it.
  assert.deepEqual(second.extras, ["note:n"]);
  assert.deepEqual(result["note:n"], [500, 100]);
  assert.deepEqual(result["group:g1"], [-20, 960]);
  const folded = (id: string) =>
    (view().nodes.find((n) => n.id === id)?.data as unknown as { folded: { id: string }[] }).folded.map((f) => f.id);
  assert.ok(folded(`super:${second.id}`).includes("note:n"));
  assert.deepEqual(result["group:g2"], [400, -40]);
});

test("a closed card re-measures its handles when its port rows change", () => {
  const src = readFileSync(new URL("../src/features/flows/editor/super-card-node.tsx", import.meta.url), "utf8");
  const view = src.slice(src.indexOf("function SuperCardViewImpl"));
  assert.match(view, /const handles = \[\.\.\.inputs, \.\.\.outputs\]\.map\(\(port\) => port\.handleId\)/);
  assert.match(view, /useEffect\(\(\) => updateNodeInternals\(id\), \[handles, id, updateNodeInternals\]\)/);
});

test("one output fanned out through folded reroutes is one card port, named by its block", () => {
  const d = doc(
    [
      { ...node("a", "input.text", [0, 0]), group: "g", label: "Script" },
      { ...node("b", "input.text", [0, 200]), group: "g" },
      node("p1", "output.preview", [900, 0]),
      node("p2", "output.preview", [900, 300]),
    ],
    [edge("e1", "a", "text", "p1", "value"), edge("e2", "a", "text", "p2", "value")],
  );
  d.groups = [{ id: "g", label: "Panel", color: "#8b5cf6", rect: [-20, -40, 600, 400] }];
  useDocStore.getState().load(d);
  useDocStore.getState().addReroute({ id: "r1", pos: [400, 20] }, "e1");
  useDocStore.getState().addReroute({ id: "r2", pos: [400, 220] }, "e2");
  select("a", "b");
  combineSelection();
  const out = view();
  const card = out.nodes.find((n) => n.id.startsWith("super:"));
  const outputs = (card?.data as unknown as { outputs: { handleId: string; label: string }[] }).outputs;
  assert.equal(outputs.length, 1);
  assert.match(outputs[0].label, /^Script · /);
  const crossing = out.edges.filter((e) => e.source === card?.id);
  assert.deepEqual(new Set(crossing.map((e) => e.sourceHandle)), new Set([outputs[0].handleId]));
  assert.deepEqual(crossing.map((e) => e.target).sort(), ["p1", "p2"]);
});

test("a note Tidy brings next to a closed card's hidden blocks stays out of the card", () => {
  const d = doc(
    [node("a", "input.text", [0, 0]), node("b", "input.text", [600, 0]), node("c", "output.preview", [1200, 0])],
    [],
  );
  d.notes = [{ id: "far", text: "about c", rect: [1490, 30, 180, 80] }];
  useDocStore.getState().load(d);
  select("a", "b");
  combineSelection();
  const [card] = cards();
  assert.deepEqual(card.extras, []);
  // Tidy up put c and its note beside the compact card, among the hidden blocks' old spread.
  useDocStore.getState().moveNodes({ c: [448, 20] });
  const { graph } = useDocStore.getState();
  useDocStore.getState().upsertNote({ ...graph.notes.far, rect: [738, 50, 180, 80] });
  const out = view();
  assert.ok(out.nodes.some((n) => n.id === "note:far"), "the note stays on the canvas");
  const folded = (out.nodes.find((n) => n.id === "super:sc1")?.data as unknown as { folded: { id: string }[] }).folded;
  assert.ok(!folded.some((f) => f.id === "note:far"));
});

test("a frame the selection box leaves out is unselected in React Flow's store too", () => {
  const src = readFileSync(new URL("../src/features/flows/editor/canvas.tsx", import.meta.url), "utf8");
  assert.match(
    src,
    /if \(change\.selected && boxSkips\(store\.getState\(\), change\.id\)\) \{[\s\S]{0,300}nodeLookup\.get\(change\.id\);\s*if \(internal\) internal\.selected = false;\s*continue;/,
  );
});
