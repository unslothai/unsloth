// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { FlowDocument } from "../src/features/flows/types.ts";
import { catalog, doc, edge, node, specs } from "./helpers/flows.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { useDocStore, documentFromState, documentSignature } = await import(
  "../src/features/flows/stores/doc-store.ts"
);
const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");
const { edgeHandles } = await import("../src/features/flows/editor/connection-rules.ts");
const {
  applySuperCards,
  boxCovers,
  foldLayout,
  isApplePlatform,
  multiSelectKeys,
  readSuperCards,
  rollUpStatus,
  selectAllIds,
  selectedBlocks,
  superCardDragMembers,
  superCardTitle,
  tidyGraph,
  toggledSelection,
} = await import("../src/features/flows/editor/super-cards.ts");
const { combineSelection, setSuperCardOpen, ungroupSuperCard } = await import(
  "../src/features/flows/editor/super-card-actions.ts"
);
const { useUiStore } = await import("../src/features/flows/stores/ui-store.ts");
const { autoLayout, rememberSizes, sizeLookup } = await import("../src/features/flows/editor/layout.ts");
const { cutSelection, groupSelection, selectAll } = await import(
  "../src/features/flows/editor/use-editor-shortcuts.ts"
);
const { tidyRequest, tidyPatch } = await import("../src/features/flows/editor/tidy.ts");

type LayoutRequest = { nodes: { id: string }[]; edges: { source: string; target: string }[] };
type Port = { handleId: string; label: string };
type Folded = { id: string; x: number; y: number };
type CardData = { memberIds: string[]; folded: Folded[]; inputs: Port[]; outputs: Port[] };
type ExtraNode = { id: string; type: string; position: { x: number; y: number }; data: Record<string, unknown> };

/** Runs the canvas's Tidy up against a stand-in layout worker and returns the request it was sent. */
async function autoLayoutRequest(): Promise<LayoutRequest> {
  const scope = globalThis as Record<string, unknown>;
  scope.document ??= { querySelector: () => null };
  scope.CSS ??= { escape: (s: string) => s };
  let sent: LayoutRequest | undefined;
  scope.Worker = class {
    onmessage: ((event: { data: unknown }) => void) | null = null;
    postMessage(message: { id: number; request: LayoutRequest }) {
      sent = message.request;
      queueMicrotask(() => this.onmessage?.({ data: { id: message.id, result: {} } }));
    }
  };
  await autoLayout(() => undefined);
  assert.ok(sent, "Tidy up sent a layout request");
  return sent;
}

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

const state = () => useDocStore.getState();
const select = (...ids: string[]) => useUiStore.getState().setSelection(ids, []);
const cards = () => readSuperCards(state().meta.editor, state().graph);

/** Selects and combines each id list in turn. */
function combine(...groups: string[][]) {
  for (const ids of groups) {
    select(...ids);
    combineSelection();
  }
}

/** Loads `d` (the comic by default) and closes `tpl` + `img` into card sc1, or the given groups. */
function loadCombined(d: FlowDocument = comic(), ...groups: string[][]) {
  useDocStore.getState().load(d);
  combine(...(groups.length ? groups : [["tpl", "img"]]));
  return state();
}

function view(extra: ExtraNode[] = []) {
  const { graph, meta } = state();
  const deriver = new RfDeriver();
  const nodes = [...deriver.nodes(graph, new Set(), null), ...extra];
  const edges = deriver.edges(graph, new Set(), () => "any");
  return applySuperCards(nodes as never, edges, graph, readSuperCards(meta.editor, graph), specs, catalog.port_colors);
}

type View = ReturnType<typeof view>;
const cardData = (out: View, id = "super:sc1") => out.nodes.find((n) => n.id === id)?.data as unknown as CardData;
const foldedIds = (out: View, id: string) => cardData(out, id).folded.map((f) => f.id);
const wireKeys = (edges: { source: string; target: string }[]) => edges.map((e) => `${e.source}>${e.target}`).sort();

/** A layout request with every block at the default size, as Tidy up sends it before folding. */
function blockRequest(withEdges = false) {
  const { graph } = state();
  return {
    direction: "LR" as const,
    nodes: graph.nodeOrder.map((id) => ({ id, width: 260, height: 140 })),
    edges: withEdges ? graph.edgeOrder.map((id) => {
      const { source, target } = graph.edges[id];
      return { source, target };
    }) : [],
  };
}
const unfoldAt = (positions: Record<string, [number, number]>) =>
  foldLayout(blockRequest(), state(), () => undefined).unfold(positions);

const tidyCtx = () => ({
  graph: state().graph,
  size: () => ({ width: 260, height: 140 }),
  portY: () => undefined,
  wireStyle: "curved" as const,
});

/** Cards as saved before they recorded their notes and reroutes: these fold by position. */
function legacyCards() {
  const { meta } = state();
  const raw = (meta.editor?.super_cards as Record<string, unknown>[]) ?? [];
  state().setMeta({ editor: { ...meta.editor, super_cards: raw.map(({ extras: _e, ...rest }) => rest) } });
}

function withoutEditor() {
  const { editor: _e, ...rest } = documentFromState(state());
  return rest;
}

const platforms: [string, string[]][] = [
  ["MacIntel", ["Meta", "Shift"]],
  ["iPad", ["Meta", "Shift"]],
  ["Win32", ["Control", "Shift"]],
  ["Linux x86_64", ["Control", "Shift"]],
  ["", ["Control", "Shift"]],
];
for (const [platform, keys] of platforms) {
  test(`Cmd adds to the selection on a Mac, Ctrl on Windows and Linux, Shift everywhere: "${platform}"`, () => {
    assert.deepEqual(multiSelectKeys(platform), keys);
  });
}

test("combining two or more selected blocks makes one closed card and selects it", () => {
  useDocStore.getState().load(comic());
  const before = withoutEditor();
  select("tpl");
  assert.equal(combineSelection(), null, "one block is not a card");
  select("tpl", "img");
  assert.equal(combineSelection(), "sc1");
  // Exactly a group's blocks: the card takes the group's name.
  assert.deepEqual(cards(), [{ id: "sc1", title: "Comic panel", members: ["tpl", "img"], extras: [] }]);
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
  loadCombined();
  select("super:sc1");
  assert.equal(combineSelection(), null, "a card alone is not a new card");
  select("super:sc1", "view");
  assert.equal(combineSelection(), "sc2");
  assert.deepEqual(cards(), [{ id: "sc2", title: "Comic panel", members: ["tpl", "img", "view"], extras: [] }]);
});

test("a selection box drawn inside a group leaves the group; one around it takes it", () => {
  const group = { x: 100, y: 100, width: 400, height: 300 };
  // Pane at translate(50, 20), zoom 0.5: flow 100 is screen 100.
  assert.equal(boxCovers({ x: 90, y: 60, width: 220, height: 170 }, [50, 20, 0.5], group), true);
  assert.equal(boxCovers({ x: 120, y: 90, width: 100, height: 100 }, [50, 20, 0.5], group), false);
});

test("a closed card can be deleted like a block", () => {
  loadCombined();
  const card = view().nodes.find((n) => n.id === "super:sc1");
  assert.ok(card);
  assert.notEqual(card.deletable, false);
});

test("a closed card shows the wires that cross into it as ports and hides its group frame", () => {
  loadCombined();
  const out = view();
  const { nodes, edges } = out;
  assert.deepEqual(nodes.map((n) => n.id).sort(), ["prompt", "raw", "super:sc1", "view"]);
  const card = nodes.find((n) => n.id === "super:sc1");
  assert.equal(card?.type, "flowSuperCard");
  assert.deepEqual(card?.position, { x: 300, y: 0 });
  const data = cardData(out);
  assert.deepEqual(data.memberIds, ["tpl", "img"]);
  assert.deepEqual(data.folded, [{ id: "group:g", x: 280, y: -40 }]);
  // The folded group's colour carries onto the card.
  assert.equal((card?.data as { color?: string }).color, "#8b5cf6");
  assert.deepEqual(data.inputs.map((p) => p.label), ["Panel script · Values"]);
  // Sorted by where the inner block sits, top first.
  assert.deepEqual(data.outputs.map((p) => p.label), ["Generate Image · Image", "Panel script · Text"]);
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
  loadCombined(d);
  assert.equal(cardData(view()).outputs.filter((p) => p.label === "Generate Image · Image").length, 1);
});

test("opening a card renders exactly as before; collapse folds it again; ungroup drops it", () => {
  useDocStore.getState().load(comic());
  const flat = view();
  combine(["tpl", "img"]);
  setSuperCardOpen("sc1", true);
  assert.equal(cards()[0].open, true);
  assert.deepEqual(useUiStore.getState().selectedNodes, ["tpl", "img"]);
  const open = view();
  assert.deepEqual(open.nodes.map((n) => n.id), flat.nodes.map((n) => n.id));
  const { graph } = state();
  assert.deepEqual(
    open.edges.map((e) => [e.id, e.source, e.sourceHandle, e.target, e.targetHandle]),
    graph.edgeOrder.map((id) => {
      const e = graph.edges[id];
      const h = edgeHandles(e);
      return [e.id, e.source, h.sourceHandle, e.target, h.targetHandle];
    }),
  );
  setSuperCardOpen("sc1", false);
  assert.ok(view().nodes.some((n) => n.id === "super:sc1"));
  ungroupSuperCard("sc1");
  assert.deepEqual(cards(), []);
  assert.equal(state().meta.editor?.super_cards, undefined);
});

test("combine, open and ungroup are edits: autosave sees them and undo/redo walk them back", () => {
  useDocStore.getState().load(comic());
  const plain = documentSignature(documentFromState(state()));
  combine(["tpl", "img"]);
  assert.notEqual(documentSignature(documentFromState(state())), plain);
  setSuperCardOpen("sc1", true);
  state().undo();
  assert.equal(cards()[0].open, undefined);
  state().undo();
  assert.deepEqual(cards(), []);
  state().redo();
  assert.deepEqual(cards().map((c) => c.id), ["sc1"]);
  state().redo();
  assert.equal(cards()[0].open, true);
});

test("flows saved with the earlier group-id format still load as cards", () => {
  const d = comic();
  d.editor = { super_cards: ["g", "gone"] };
  useDocStore.getState().load(d);
  assert.deepEqual(cards(), [{ id: "g", title: "Comic panel", members: ["tpl", "img"] }]);
  assert.ok(view().nodes.some((n) => n.id === "super:g"));
});

test("two closed cards wire card to card", () => {
  loadCombined(comic(), ["tpl", "img"], ["view", "raw"]);
  const e3 = view().edges.find((e) => e.id === "e3");
  assert.equal(e3?.source, "super:sc1");
  assert.equal(e3?.target, "super:sc2");
});

test("dragging a card carries its blocks and its folded group frame", () => {
  const { graph } = loadCombined();
  const card = view().nodes.find((n) => n.id === "super:sc1");
  assert.deepEqual(Object.fromEntries(superCardDragMembers(card as never, graph) ?? []), {
    tpl: { x: 300, y: 40 },
    img: { x: 600, y: 0 },
    "group:g": { x: 280, y: -40 },
  });
});

const rollUps: [string, Parameters<typeof rollUpStatus>[0], string][] = [
  ["nothing has run", [undefined, undefined], "idle"],
  ["one done, one queued", ["done", "queued"], "running"],
  ["queued and idle", ["queued", "idle"], "queued"],
  ["one failed", ["done", "failed"], "failed"],
  ["done and cached", ["done", "cached"], "done"],
  ["only skipped", ["skipped"], "idle"],
  // A recipe's columns stay idle while its Recipe Dataset step runs and finishes.
  ["recipe columns idle beside a finished step", [undefined, "idle", "done"], "done"],
  ["one cancelled", ["done", "cancelled"], "cancelled"],
];
for (const [name, statuses, status] of rollUps) {
  test(`the card rolls its blocks' run status into one: ${name}`, () => {
    assert.equal(rollUpStatus(statuses).status, status);
  });
}

test("the card rolls its blocks' run status into one: running counts done of total", () => {
  assert.deepEqual(rollUpStatus(["done", "running", undefined]), { status: "running", done: 1, total: 2 });
});

test("a recipe group's card is titled as a recipe", () => {
  assert.equal(superCardTitle({ label: "Text to SQL", kind: "recipe" }), "Recipe: Text to SQL");
  assert.equal(superCardTitle({ label: "Comic panel" }), "Comic panel");
});

test("auto layout places a closed card as one block and moves its blocks with it", () => {
  const d = comic();
  d.notes = [
    { id: "n1", text: "inside", rect: [320, 100, 100, 60] },
    { id: "n2", text: "outside", rect: [0, 900, 100, 60] },
  ];
  loadCombined(d);
  const folded = foldLayout(blockRequest(true), state(), () => ({ width: 288, height: 120 }));
  assert.deepEqual(folded.request.nodes.map((n) => n.id).sort(), ["prompt", "raw", "super:sc1", "view"]);
  assert.deepEqual(wireKeys(folded.request.edges), ["prompt>super:sc1", "super:sc1>raw", "super:sc1>view"]);
  const result = folded.unfold({ "super:sc1": [380, 60], prompt: [0, 0] });
  assert.equal(result["super:sc1"], undefined);
  assert.deepEqual(result.tpl, [380, 100]);
  assert.deepEqual(result.img, [680, 60]);
  // The folded group frame and the note among the blocks move by the same amount.
  assert.deepEqual(result["group:g"], [360, 20]);
  assert.deepEqual(result["note:n1"], [400, 160]);
  assert.equal(result["note:n2"], undefined);
});

test("tidy up folds a group the card holds whole, so the card keeps its wires", () => {
  loadCombined();
  const request = tidyRequest(tidyCtx());
  assert.ok(request.nodes.some((n) => n.id === "group:g"));
  const folded = foldLayout(request, state(), () => undefined);
  assert.deepEqual(folded.request.nodes.map((n) => n.id).sort(), ["prompt", "raw", "super:sc1", "view"]);
  assert.deepEqual(wireKeys(folded.request.edges), ["prompt>super:sc1", "super:sc1>raw", "super:sc1>view"]);
});

test("auto layout moves a note a wide block folded, after the block left React Flow", () => {
  const d = doc([node("a", "input.text", [0, 0]), node("b", "image.generate", [300, 0])]);
  // Centred inside b's measured 500x400 box, outside the 260x140 default.
  d.notes = [{ id: "n1", text: "beside", rect: [700, 250, 100, 60] }];
  useDocStore.getState().load(d);
  // The mounted canvas has measured b by the time the card closes.
  rememberSizes(new Map([["b", { width: 500, height: 400 }]]));
  combine(["a", "b"]);
  // A closed card's blocks are not in React Flow, so it reports no size for them.
  const lookup = sizeLookup(() => undefined);
  const result = foldLayout(blockRequest(), state(), lookup).unfold({ "super:sc1": [100, 50] });
  assert.deepEqual(result.b, [400, 50]);
  assert.deepEqual(result["note:n1"], [800, 300]);
  // The sizes belong to the mounted canvas: another flow's canvas, or none, brings its own.
  rememberSizes(new Map([["a", { width: 260, height: 140 }]]));
  assert.equal(lookup("b"), undefined);
  rememberSizes(new Map());
  assert.equal(lookup("a"), undefined);
});

test("notes and reroutes drawn among a card's blocks fold with it", () => {
  loadCombined();
  legacyCards();
  const { graph, meta } = state();
  const deriver = new RfDeriver();
  const note = (id: string, x: number, y: number) => ({
    id: `note:${id}`, type: "flowNote", position: { x, y }, data: { noteId: id },
  });
  const nodes = [
    ...deriver.nodes(graph, new Set(), null),
    { id: "rr1", type: "flowReroute", position: { x: 500, y: 100 }, data: { nodeId: "rr1" } },
    note("n1", 320, 100),
    note("n3", 700, 220),
    note("n2", 300, 900),
  ];
  const edges = [
    ...deriver.edges(graph, new Set(), () => "any"),
    { id: "w1", source: "prompt", target: "rr1", sourceHandle: "o:text", targetHandle: "in" },
  ];
  const out = applySuperCards(nodes as never, edges, graph, readSuperCards(meta.editor, graph), specs);
  assert.ok(!out.nodes.some((n) => n.id === "rr1" || n.id === "note:n1"));
  // Inside the folded group frame, below the blocks.
  assert.ok(!out.nodes.some((n) => n.id === "note:n3"));
  assert.ok(out.nodes.some((n) => n.id === "note:n2"));
  // Dragging the card carries them, where they sit.
  const card = out.nodes.find((n) => n.id === "super:sc1");
  const members = superCardDragMembers(card as never, graph);
  assert.deepEqual(members?.get("rr1"), { x: 500, y: 100 });
  assert.deepEqual(members?.get("note:n1"), { x: 320, y: 100 });
  assert.equal(out.edges.find((e) => e.id === "w1")?.target, "super:sc1");
});

test("a note in one card's folded frame stays with that card where another card's blocks overlap it", () => {
  loadCombined(comic(), ["tpl", "img"], ["view", "raw"]);
  legacyCards();
  const out = view([{ id: "note:edge", type: "flowNote", position: { x: 910, y: 100 }, data: { noteId: "edge" } }]);
  assert.ok(foldedIds(out, "super:sc1").includes("note:edge"));
  assert.ok(!foldedIds(out, "super:sc2").includes("note:edge"));
});

test("a Mac's Ctrl-click (a right-click to the browser) adds or drops a card from the selection", () => {
  assert.deepEqual(toggledSelection(["a"], "b"), ["a", "b"]);
  assert.deepEqual(toggledSelection(["a", "b"], "a"), ["b"]);
  assert.deepEqual(toggledSelection([], "a"), ["a"]);
  assert.equal(isApplePlatform("MacIntel"), true);
  assert.equal(isApplePlatform("Win32"), false);
});

test("copy and duplicate take the blocks inside a selected closed card", () => {
  const { graph, meta } = loadCombined();
  assert.deepEqual(selectedBlocks(["super:sc1", "view", "gone"], meta.editor, graph), ["tpl", "img", "view"]);
  // Cut removes what copy took, not the card's own selection id.
  select("super:sc1", "view");
  const cut = cutSelection();
  assert.deepEqual(cut?.nodes.map((n) => n.id), ["tpl", "img", "view"]);
  assert.deepEqual(cut?.edges.map((e) => e.id), ["e2", "e3"]);
  assert.deepEqual(state().graph.nodeOrder, ["prompt", "raw"]);
  assert.deepEqual(useUiStore.getState().selectedNodes, []);
});

test("a block added where a folded frame sits stays out of it", () => {
  const doc0 = loadCombined();
  doc0.addNode({ ...node("new", "output.preview", [400, 100]) });
  doc0.insertFragment({ nodes: [node("pasted", "output.preview", [420, 120])], edges: [] });
  const { graph } = state();
  assert.equal(graph.nodes.new.group ?? null, null);
  assert.equal(graph.nodes.pasted.group ?? null, null);
  // An open frame still takes what lands in it.
  setSuperCardOpen("sc1", true);
  state().addNode({ ...node("inside", "output.preview", [400, 100]) });
  assert.equal(state().graph.nodes.inside.group, "g");
});

test("tidy up keeps a card's blocks together when the card holds part of a group", async () => {
  const d = comic();
  d.notes = [{ id: "n1", text: "by the template", rect: [320, 100, 100, 60] }];
  // tpl is in group g with img; raw is outside it.
  const s = loadCombined(d, ["tpl", "raw"]);
  const ctx = tidyCtx();
  const folded = foldLayout(tidyRequest({ ...ctx, graph: tidyGraph(s.graph, s.meta.editor) }), s, () => undefined);
  // tpl's wires (prompt -> tpl, tpl -> img, tpl -> raw) reach the card, not group g's placeholder.
  const wires = wireKeys(folded.request.edges);
  assert.deepEqual(wires, ["group:g>view", "prompt>super:sc1", "super:sc1>group:g"]);
  // Tidy up on the canvas sends the layout worker the same folded request.
  assert.deepEqual(wireKeys((await autoLayoutRequest()).edges), wires);
  const result = folded.unfold({ "super:sc1": [1000, 500], "group:g": [0, 0], prompt: [0, 0], view: [0, 0] });
  const patch = tidyPatch(ctx, result);
  const [tx, ty] = patch.nodes.tpl.pos;
  const [rx, ry] = patch.nodes.raw.pos;
  assert.deepEqual([rx - tx, ry - ty], [900 - 300, 300 - 40]);
  // A note folded into the card moves with the card, not with the nearest loose block.
  const [nx, ny] = patch.notes.n1;
  assert.deepEqual([nx - tx, ny - ty], [320 - 300, 100 - 40]);
});

test("tidy up moves a frame whose blocks all went into different closed cards", () => {
  // Group g holds tpl and img, and each lands in its own closed card.
  const s = loadCombined(comic(), ["tpl", "raw"], ["img", "view"]);
  const ctx = tidyCtx();
  const folded = foldLayout(tidyRequest({ ...ctx, graph: tidyGraph(s.graph, s.meta.editor) }), s, () => undefined);
  assert.ok(!folded.request.nodes.some((n) => n.id === "group:g"));
  const ids = cards().map((c) => `super:${c.id}`);
  const patch = tidyPatch(ctx, folded.unfold({ prompt: [0, 0], [ids[0]]: [2000, 2000], [ids[1]]: [2400, 2000] }));
  // The frame follows its blocks instead of staying where they were.
  const [tx, ty] = patch.nodes.tpl.pos;
  const [gx, gy] = patch.groups.g;
  assert.deepEqual([gx - tx, gy - ty], [280 - 300, -40 - 40]);
});

test("Select All takes closed cards too, so Delete removes them", () => {
  const { meta, graph } = loadCombined();
  const all = selectAllIds(meta.editor, graph);
  assert.ok(all.includes("super:sc1") && all.includes("prompt"));
  selectAll();
  assert.deepEqual(useUiStore.getState().selectedNodes, all);
  assert.deepEqual(useUiStore.getState().selectedEdges, graph.edgeOrder);
});

test("a note inside one card's folded frame and another card's blocks moves only with the frame's card", () => {
  const d = comic();
  d.notes = [{ id: "n1", text: "both", rect: [880, 70, 60, 60] }];
  loadCombined(d, ["tpl", "img"], ["view", "raw"]);
  // sc1 moves by (100, 0), sc2 by (0, 500).
  const result = unfoldAt({ "super:sc1": [400, -0], "super:sc2": [900, 500] });
  assert.deepEqual(result["note:n1"], [980, 70]);
});

test("a free reroute folded into a card moves with it on Tidy", () => {
  const d = comic();
  d.editor = { reroutes: [{ id: "rr1", pos: [450, 60] }] };
  loadCombined(d);
  const result = unfoldAt({ "super:sc1": [400, 100] });
  assert.deepEqual(result.rr1, [550, 160]);
  assert.deepEqual(tidyPatch(tidyCtx(), result).reroutes.rr1, [550, 160]);
});

test("resizing a visible frame never hands its blocks to a folded one", () => {
  const d = comic();
  d.groups = [...(d.groups ?? []), { id: "outer", label: "Outer", color: "#64748b", rect: [-100, -200, 1500, 900] }];
  d.nodes = [...d.nodes, { ...node("x", "output.preview", [420, 60]), group: "outer" }];
  loadCombined(d);
  state().resizeGroup({ ...state().graph.groups.outer, rect: [-100, -200, 1520, 900] });
  const { graph } = state();
  assert.equal(graph.nodes.x.group, "outer");
  assert.equal(graph.nodes.tpl.group, "g");
});

test("an unmeasured note folds by its saved size, not its corner", () => {
  const d = comic();
  // Its corner is among the card's blocks; its middle is well outside them.
  d.notes = [{ id: "big", text: "big", rect: [320, 100, 800, 600] }];
  loadCombined(d);
  assert.ok(view().nodes.some((n) => n.id === "note:big"));
});

test("after a reload a closed card folds by its blocks' saved sizes", () => {
  const sized = { ...node("b", "image.generate", [300, 0]), size: [500, 400] as [number, number] };
  const d = doc([node("a", "input.text", [0, 0]), sized]);
  d.notes = [{ id: "n1", text: "beside", rect: [700, 250, 100, 60] }];
  loadCombined(d, ["a", "b"]);
  rememberSizes(new Map());
  assert.ok(!view().nodes.some((n) => n.id === "note:n1"), "the note folds into the card");
  assert.deepEqual(unfoldAt({ "super:sc1": [100, 50] })["note:n1"], [800, 300]);
});

test("grouping a selection with a closed card groups the card's blocks", () => {
  const { meta, graph } = loadCombined();
  assert.deepEqual(selectedBlocks(["super:sc1", "view"], meta.editor, graph), ["tpl", "img", "view"]);
  select("super:sc1", "view");
  groupSelection();
  const after = state().graph;
  const made = Object.keys(after.groups).find((id) => id !== "g");
  assert.ok(made);
  assert.deepEqual(after.nodeOrder.filter((id) => after.nodes[id].group === made), ["tpl", "img", "view"]);
});

test("a rebuilt closed card keeps its measurement, so React Flow does not hide it to measure again", () => {
  const { graph, meta } = loadCombined();
  const deriver = new RfDeriver();
  const out = applySuperCards(
    deriver.nodes(graph, new Set(), null) as never,
    deriver.edges(graph, new Set(), () => "any"),
    graph,
    readSuperCards(meta.editor, graph),
    specs,
    catalog.port_colors,
    new Set(),
    (id) => (id === "super:sc1" ? { width: 300, height: 180 } : undefined),
  );
  assert.deepEqual(out.nodes.find((n) => n.id === "super:sc1")?.measured, { width: 300, height: 180 });
});

test("resizing a frame leaves a block hidden in a closed card in its frame, even when the card holds only part of it", () => {
  const d = comic();
  // The card was dragged, so its hidden block's saved spot is now outside the frame.
  d.nodes = d.nodes.map((n) => (n.id === "tpl" ? { ...n, pos: [-600, 500] as [number, number] } : n));
  loadCombined(d, ["tpl", "prompt"]);
  const g = state().graph.groups.g;
  state().resizeGroup({ ...g, rect: [g.rect[0], g.rect[1], g.rect[2] + 10, g.rect[3]] });
  assert.equal(state().graph.nodes.tpl.group, "g");
  assert.equal(state().graph.nodes.img.group, "g");
});

test("a note where two folded frames overlap moves with the card the canvas folds it into", () => {
  const d = doc([
    { ...node("a", "input.text", [0, 0]), group: "g1" },
    { ...node("a2", "input.text", [0, 200]), group: "g1" },
    { ...node("b", "input.text", [1000, 0]), group: "g2" },
    { ...node("b2", "input.text", [1000, 200]), group: "g2" },
  ]);
  d.groups = [
    { id: "g1", label: "One", color: "#8b5cf6", rect: [-20, -40, 700, 400] },
    { id: "g2", label: "Two", color: "#8b5cf6", rect: [400, -40, 900, 400] },
  ];
  d.notes = [{ id: "n", text: "both", rect: [500, 100, 100, 50] }];
  // The second frame's card is made first, so card order and block order disagree.
  loadCombined(d, ["b", "b2"], ["a", "a2"]);
  const [second, first] = cards();
  assert.deepEqual(first.members, ["a", "a2"]);
  assert.deepEqual(second.members, ["b", "b2"]);
  const result = unfoldAt({ [`super:${first.id}`]: [0, 1000], [`super:${second.id}`]: [1000, 0] });
  // The note was folded into the card made first, and moves with that card, as the canvas shows it.
  assert.deepEqual(second.extras, ["note:n"]);
  assert.deepEqual(result["note:n"], [500, 100]);
  assert.deepEqual(result["group:g1"], [-20, 960]);
  assert.ok(foldedIds(view(), `super:${second.id}`).includes("note:n"));
  assert.deepEqual(result["group:g2"], [400, -40]);
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
  state().addReroute({ id: "r1", pos: [400, 20] }, "e1");
  state().addReroute({ id: "r2", pos: [400, 220] }, "e2");
  combine(["a", "b"]);
  const out = view();
  const card = out.nodes.find((n) => n.id.startsWith("super:"));
  const { outputs } = cardData(out, card?.id);
  assert.equal(outputs.length, 1);
  assert.match(outputs[0].label, /^Script · /);
  const crossing = out.edges.filter((e) => e.source === card?.id);
  assert.deepEqual(new Set(crossing.map((e) => e.sourceHandle)), new Set([outputs[0].handleId]));
  assert.deepEqual(crossing.map((e) => e.target).sort(), ["p1", "p2"]);
});

test("a note Tidy brings next to a closed card's hidden blocks stays out of the card", () => {
  const d = doc([node("a", "input.text", [0, 0]), node("b", "input.text", [600, 0])]);
  d.nodes.push(node("c", "output.preview", [1200, 0]));
  d.notes = [{ id: "far", text: "about c", rect: [1490, 30, 180, 80] }];
  loadCombined(d, ["a", "b"]);
  const [card] = cards();
  assert.deepEqual(card.extras, []);
  // Tidy up put c and its note beside the compact card, among the hidden blocks' old spread.
  state().moveNodes({ c: [448, 20] });
  state().upsertNote({ ...state().graph.notes.far, rect: [738, 50, 180, 80] });
  const out = view();
  assert.ok(out.nodes.some((n) => n.id === "note:far"), "the note stays on the canvas");
  assert.ok(!foldedIds(out, "super:sc1").includes("note:far"));
});
