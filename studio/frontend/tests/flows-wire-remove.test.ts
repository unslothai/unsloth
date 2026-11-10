// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A wire can be removed by selecting it, by its × button, or by dragging its end off the port.

import assert from "node:assert/strict";
import test from "node:test";

import { doc, edge, node, specs } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useCatalogStore } = await import("../src/features/flows/stores/catalog-store.ts");
useCatalogStore.setState({ specs });
const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { inHandle, outHandle } = await import("../src/features/flows/editor/connection-rules.ts");
const { dropRemovesWire, reconnectWire, wireEdgeOptions } = await import(
  "../src/features/flows/editor/use-wire-reconnect.ts"
);
const { wireDeleteShown } = await import("../src/features/flows/editor/wire-live.ts");
const { RfDeriver } = await import("../src/features/flows/editor/derive-rf.ts");

const fanIn = () =>
  useDocStore.getState().load(
    doc(
      [node("a", "input.text"), node("b", "input.text", [0, 200]), node("p", "output.preview", [500, 100]), node("q", "output.preview", [500, 400])],
      [edge("e1", "a", "text", "p", "value"), edge("e2", "b", "text", "p", "value")],
    ),
  );
const rf = (id: string, source: string, target: string) => ({
  id,
  source,
  target,
  sourceHandle: outHandle("text"),
  targetHandle: inHandle("value"),
});

test("removing one wire of a fan-in leaves the other, as one undo step", () => {
  fanIn();
  const depth = useDocStore.getState().past.length;
  useDocStore.getState().removeElements({ edges: ["e1"] });
  assert.deepEqual(useDocStore.getState().graph.edgeOrder, ["e2"]);
  assert.equal(useDocStore.getState().past.length, depth + 1);
  useDocStore.getState().undo();
  assert.deepEqual(useDocStore.getState().graph.edgeOrder, ["e1", "e2"]);
});

test("dragging a wire's end to another input moves it in one undo step", () => {
  fanIn();
  const depth = useDocStore.getState().past.length;
  const error = reconnectWire(rf("e1", "a", "p") as never, { source: "a", sourceHandle: outHandle("text"), target: "q", targetHandle: inHandle("value") });
  assert.equal(error, null);
  const g = useDocStore.getState().graph;
  assert.equal(g.edges.e1, undefined);
  assert.ok(Object.values(g.edges).some((e) => e.source === "a" && e.target === "q"));
  assert.equal(useDocStore.getState().past.length, depth + 1, "one history entry");
  useDocStore.getState().undo();
  assert.deepEqual(useDocStore.getState().graph.edgeOrder, ["e1", "e2"]);
});

test("a wire moved with a full undo history is still one undo step", async () => {
  const { HISTORY_LIMIT } = await import("../src/features/flows/stores/doc-store.ts");
  fanIn();
  for (let i = 0; i < HISTORY_LIMIT; i++) useDocStore.getState().moveNodes({ a: [i + 1, 0] });
  const moved = useDocStore.getState().graph;
  assert.equal(useDocStore.getState().past.length, HISTORY_LIMIT);
  reconnectWire(rf("e1", "a", "p") as never, { source: "a", sourceHandle: outHandle("text"), target: "q", targetHandle: inHandle("value") });
  useDocStore.getState().undo();
  assert.deepEqual(useDocStore.getState().graph.edgeOrder, moved.edgeOrder);
  assert.equal(useDocStore.getState().graph.edges.e1?.target, "p");
});

test("a refused move puts the wire back untouched", () => {
  fanIn();
  const before = useDocStore.getState().graph;
  const depth = useDocStore.getState().past.length;
  const error = reconnectWire(rf("e1", "a", "p") as never, { source: "a", sourceHandle: outHandle("text"), target: "a", targetHandle: inHandle("nope") });
  assert.ok(error);
  assert.equal(useDocStore.getState().graph, before);
  assert.equal(useDocStore.getState().past.length, depth);
});

const drops: [string, boolean, unknown, boolean][] = [
  ["on empty canvas removes the wire", false, null, true],
  ["on a port that refused it keeps the wire", false, { id: "in:value" }, false],
  ["after moving it to another port keeps the moved wire", true, { id: "in:value" }, false],
  ["after a move, off any port, keeps the moved wire", true, null, false],
];
for (const [name, moved, toHandle, removes] of drops) {
  test(`a dragged wire end let go ${name}`, () => {
    assert.equal(dropRemovesWire(moved, toHandle), removes);
  });
}

test("only a wire's input end drags; a read-only canvas gives no × and no drag-off", () => {
  assert.deepEqual(wireEdgeOptions(true), { reconnectable: "target" });
  assert.deepEqual(wireEdgeOptions(false), { reconnectable: false, deletable: false });
});

const shown: [string, boolean | undefined, boolean, boolean | undefined, boolean][] = [
  ["hovered", undefined, true, false, true],
  ["selected", true, false, true, true],
  ["neither hovered nor selected", true, false, false, false],
  ["locked, though hovered and selected", false, true, true, false],
];
for (const [name, deletable, hover, selected, visible] of shown) {
  test(`a wire's × button: ${name} -> ${visible ? "shown" : "hidden"}`, () => {
    assert.equal(wireDeleteShown(deletable, hover, selected), visible);
  });
}

test("a reroute's wire segments have no draggable end; they move with the reroute", () => {
  fanIn();
  useDocStore.getState().addReroute({ id: "r1", pos: [250, 0] }, "e1");
  const before = useDocStore.getState().graph;
  const segments = Object.keys((before as { wires?: Record<string, unknown> }).wires ?? {});
  assert.ok(segments.length > 0, "the reroute split e1 into segments");
  const rf = new RfDeriver().edges(before, new Set(), () => "any");
  for (const id of segments)
    assert.equal(rf.find((e) => e.id === id)?.reconnectable, false, `${id} is not reconnectable`);
  // Even when React Flow offers a segment's end, the move is refused and nothing changes; this
  // segment keeps the wire's own id, so it is a graph edge too.
  const depth = useDocStore.getState().past.length;
  const segment = rf.find((e) => e.id === "e1");
  assert.ok(segments.includes("e1") && before.edges.e1);
  assert.ok(segment);
  assert.equal(
    reconnectWire(segment, { source: "a", sourceHandle: outHandle("text"), target: "q", targetHandle: inHandle("value") }),
    null,
  );
  assert.equal(useDocStore.getState().graph, before);
  assert.equal(useDocStore.getState().past.length, depth);
});
