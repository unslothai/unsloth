// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A wire can be removed by selecting it, by its × button, or by dragging its end off the port.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { doc, edge, node } from "./flows-editor-fixtures.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const catalog = JSON.parse(readFileSync(new URL("./fixtures/flows-catalog.json", import.meta.url), "utf8"));
const { useCatalogStore } = await import("../src/features/flows/stores/catalog-store.ts");
useCatalogStore.setState({ specs: Object.fromEntries(catalog.nodes.map((s: { type: string }) => [s.type, s])) });
const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { inHandle, outHandle } = await import("../src/features/flows/editor/connection-rules.ts");
const { reconnectWire } = await import("../src/features/flows/editor/use-wire-reconnect.ts");

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

test("a refused move puts the wire back untouched", () => {
  fanIn();
  const before = useDocStore.getState().graph;
  const depth = useDocStore.getState().past.length;
  const error = reconnectWire(rf("e1", "a", "p") as never, { source: "a", sourceHandle: outHandle("text"), target: "a", targetHandle: inHandle("nope") });
  assert.ok(error);
  assert.equal(useDocStore.getState().graph, before);
  assert.equal(useDocStore.getState().past.length, depth);
});

test("drag-off removes through the canvas delete path; inputs are the reconnectable end", () => {
  const hook = readSrc("features/flows/editor/use-wire-reconnect.ts");
  assert.match(hook, /if \(!moved\.current\) void deleteElements\(\{ edges: \[\{ id: edge\.id \}\] \}\)/);
  assert.match(hook, /reconnectable: "target"/);
  assert.match(readSrc("features/flows/editor/canvas.tsx"), /\{\.\.\.wireReconnect\}/);
});

test("each wire draws its own × at its midpoint on hover or selection, and a selected wire gets a halo", () => {
  const del = readSrc("features/flows/editor/wire-delete.tsx");
  assert.match(del, /deleteElements\(\{ edges: \[\{ id \}\] \}\)/);
  assert.match(del, /aria-label="Remove wire"/);
  const curved = readSrc("features/flows/editor/flow-edge.tsx");
  assert.match(curved, /<WireDelete[\s\S]*show=\{deletable !== false && \(hover \|\| !!selected\)\}/, "no × on a wire that cannot be removed");
  assert.match(curved, /\{selected && <WireHalo path=\{path\} \/>\}/);
});
