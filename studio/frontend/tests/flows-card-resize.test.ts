// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Cards resize from any edge or corner, live while dragging, saved in the flow as one undo step.

import assert from "node:assert/strict";
import test from "node:test";

import { doc, node } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore, documentFromState } = await import("../src/features/flows/stores/doc-store.ts");
const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");
const { CARD_MIN_WIDTH, cardMinHeight } = await import("../src/features/flows/editor/card-size.ts");
const { clampFitToFrame } = await import("../src/features/flows/editor/layout.ts");

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
});

test("a card's floor covers its header, ports, fields and error; only the output preview gives way", async () => {
  const { cardContentMinHeight } = await import("../src/features/flows/editor/card-size.ts");
  type Fake = { offsetHeight: number; clientHeight?: number; children: Fake[]; attrs: string[]; style?: Record<string, string> };
  const el = (offsetHeight: number, attrs: string[] = [], children: Fake[] = [], style: Record<string, string> = {}): Fake =>
    ({ offsetHeight, children, attrs, style });
  const body = el(999, ["data-card-body"], [
    el(48), // ports
    el(22), // BIOME field row
    el(24), // "Show inputs"
    el(400, ["data-card-grow"]), // output preview
    el(52), // error
  ], { paddingTop: "8px", paddingBottom: "8px", rowGap: "6px" });
  const cardEl: Fake = {
    offsetHeight: 300,
    clientHeight: 298, // 1px border top and bottom
    attrs: [],
    children: [el(8, [], [], { position: "absolute" }), el(64), body],
  };
  const asDom = (f: Fake): unknown => ({
    offsetHeight: f.offsetHeight,
    clientHeight: f.clientHeight ?? f.offsetHeight,
    children: f.children.map(asDom),
    hasAttribute: (a: string) => f.attrs.includes(a),
    _style: f.style ?? {},
  });
  const styleOf = (e: Element) => {
    const s = (e as unknown as { _style: Record<string, string> })._style;
    return { position: s.position ?? "static", rowGap: s.rowGap ?? "0", paddingTop: s.paddingTop ?? "0", paddingBottom: s.paddingBottom ?? "0" };
  };
  // border 2 + header 64 + padding 16 + 4 gaps 24 + ports/field/disclosure/error 146 + preview floor 24
  assert.equal(cardContentMinHeight(asDom(cardEl) as HTMLElement, styleOf as never), 276);
});

// Frame [0, 0, 500, 400] with pad 12: a card at (100, 100) may grow to 388 x 288 inside it.
for (const [why, fit, size, out] of [
  ["growth stops at the frame edge", [600, 500], { width: 200, height: 150 }, [388, 288]],
  ["a fit inside the frame is kept", [300, 200], { width: 200, height: 150 }, [300, 200]],
  ["a card already past the frame keeps its size", [700, 600], { width: 450, height: 350 }, [450, 350]],
  ["a card past the frame can still shrink to its fit", [420, 320], { width: 450, height: 350 }, [420, 320]],
] as const)
  test(`Tidy's frame clamp only limits growth: ${why}`, () => {
    assert.deepEqual(clampFitToFrame([...fit], size, [100, 100], [0, 0, 500, 400]), out);
  });
