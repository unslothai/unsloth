// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Cards resize from any edge or corner, live while dragging, saved in the flow as one undo step.

import assert from "node:assert/strict";
import test from "node:test";

import { doc, node } from "./flows-editor-fixtures.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore, documentFromState } = await import("../src/features/flows/stores/doc-store.ts");
const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");
const { CARD_MIN_WIDTH, cardMinHeight } = await import("../src/features/flows/editor/card-size.ts");

const card = (id: string, boxes?: Record<string, { width: number; height: number }>) =>
  new RfDeriver()
    .nodes(useDocStore.getState().graph, new Set(), null, undefined, boxes)
    .find((n) => n.id === id);

test("a flow saved before resizing existed keeps the automatic size", () => {
  useDocStore.getState().load(doc([node("a", "output.preview")], []));
  assert.equal(useDocStore.getState().graph.nodes.a.size, undefined);
  assert.equal(card("a")?.style, undefined);
});

test("a resize is one undo step and survives a save and reload", () => {
  useDocStore.getState().load(doc([node("a", "output.preview", [100, 100])], []));
  const depth = useDocStore.getState().past.length;
  useDocStore.getState().updateNode("a", { pos: [80, 90], size: [420, 360] });
  assert.equal(useDocStore.getState().past.length, depth + 1);
  assert.deepEqual(card("a")?.style, { width: 420, height: 360 });
  assert.deepEqual(card("a")?.position, { x: 80, y: 90 });

  const saved = JSON.parse(JSON.stringify(documentFromState(useDocStore.getState())));
  useDocStore.getState().load(saved);
  assert.deepEqual(useDocStore.getState().graph.nodes.a.size, [420, 360]);

  useDocStore.getState().updateNode("a", { pos: [0, 0], size: [500, 400] });
  useDocStore.getState().undo();
  assert.deepEqual(useDocStore.getState().graph.nodes.a.size, [420, 360]);
  assert.deepEqual(useDocStore.getState().graph.nodes.a.pos, [80, 90]);
});

test("a live box draws under the handle before anything is committed, for cards and notes", () => {
  const d = doc([node("a", "llm.generate")], []);
  d.notes = [{ id: "n1", text: "", rect: [0, 300, 240, 140] }];
  useDocStore.getState().load(d);
  const writes = useDocStore.getState().past.length;
  const boxes = { a: { width: 400, height: 300 }, "note:n1": { width: 380, height: 220 } };
  const derived = new RfDeriver().nodes(useDocStore.getState().graph, new Set(), null, undefined, boxes);
  assert.deepEqual(derived.find((n) => n.id === "a")?.style, boxes.a);
  assert.deepEqual(derived.find((n) => n.id === "note:n1")?.style, boxes["note:n1"]);
  assert.equal(useDocStore.getState().past.length, writes, "no store write while resizing");
});

test("a collapsed card ignores its stored size", () => {
  useDocStore.getState().load(doc([{ ...node("a", "output.preview"), size: [420, 360], collapsed: true }], []));
  assert.equal(card("a")?.style, undefined);
});

test("the minimum size keeps the header and every port row", () => {
  const spec = { inputs: [{}, {}, {}], outputs: [{}], flow_out: ["next"] } as never;
  assert.equal(cardMinHeight(spec), 56 + 3 * 24 + 16);
  assert.ok(CARD_MIN_WIDTH >= 200);
  const src = readSrc("features/flows/editor/flow-node.tsx");
  assert.match(src, /minWidth: CARD_MIN_WIDTH, minHeight, onResizeEnd/);
  assert.match(src, /<CardResizer id=\{id\} minHeight=\{cardMinHeight\(spec\)\} \/>/);
});

test("the canvas draws resizes live and commits nothing until release", () => {
  const src = readSrc("features/flows/editor/canvas.tsx");
  const handler = src.slice(src.indexOf("const onNodesChange"), src.indexOf("const onEdgesChange"));
  assert.match(handler, /change\.type === "dimensions" && change\.resizing/);
  assert.doesNotMatch(handler, /useDocStore/, "no store write per frame");
});

test("a drag of several cards is one store write and one undo step", () => {
  useDocStore.getState().load(doc([node("a", "input.text"), node("b", "input.text", [300, 0]), node("c", "input.text", [600, 0])], []));
  let writes = 0;
  const off = useDocStore.subscribe((s, p) => { if (s.graph !== p.graph) writes++; });
  const depth = useDocStore.getState().past.length;
  useDocStore.getState().moveNodes({ a: [10, 10], b: [310, 10], c: [610, 10] });
  off();
  assert.equal(writes, 1);
  assert.equal(useDocStore.getState().past.length, depth + 1);
  const src = readSrc("features/flows/editor/canvas.tsx");
  const stop = src.slice(src.indexOf("const onNodeDragStop"));
  assert.equal((stop.slice(0, stop.indexOf("const onConnect")).match(/\.moveNodes\(/g) ?? []).length, 1);
});

test("a resized card clips its content top to bottom only, so the port dots on its edges stay whole", () => {
  const view = readSrc("features/flows/editor/flow-node.tsx");
  assert.match(view, /group-data-\[sized\]\/node:\[overflow-y:clip\]/);
  assert.doesNotMatch(view, /group-data-\[sized\]\/node:overflow-(hidden|clip)\b/);
  assert.match(readSrc("features/flows/editor/node-card.tsx"), /corner-squircle group\/node min-w-0 overflow-visible/);
});
