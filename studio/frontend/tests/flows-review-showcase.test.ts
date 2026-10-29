// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Fixes from the train-logic showcase: rewired cards drop their old output, Escape closes the
// inspector, fit keeps text readable, and a resized card writes out lists and objects in full.

import assert from "node:assert/strict";
import test from "node:test";

import { doc, edge, node } from "./flows-editor-fixtures.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { useRuntimeStore } = await import("../src/features/flows/stores/runtime-store.ts");
const { rewiredCards, watchRewiredOutputs } = await import("../src/features/flows/editor/stale-outputs.ts");

const flow = () =>
  doc(
    [node("a", "input.text"), node("b", "input.text", [0, 200]), node("p", "output.preview", [500, 0])],
    [edge("e1", "a", "text", "p", "value")],
  );

test("rewiring a card's input drops the output it showed; a card mid-run keeps its own", () => {
  useDocStore.getState().load(flow());
  const stop = watchRewiredOutputs();
  useRuntimeStore.getState().seedOutputs({ p: { status: "done", outputs: { value: "old" } }, a: { status: "done" } });
  useDocStore.getState().removeElements({ edges: ["e1"] });
  useDocStore.getState().connect(edge("e2", "b", "text", "p", "value"), true);
  assert.equal(useRuntimeStore.getState().nodes.p, undefined, "p no longer shows the old run");
  assert.ok(useRuntimeStore.getState().nodes.a, "a was not rewired");
  useRuntimeStore.getState().seedOutputs({ p: { status: "running" } });
  useDocStore.getState().removeElements({ edges: ["e2"] });
  assert.equal(useRuntimeStore.getState().nodes.p?.status, "running");
  stop();
});

test("only cards whose incoming wires changed count as rewired", () => {
  useDocStore.getState().load(flow());
  const before = useDocStore.getState().graph;
  useDocStore.getState().updateNode("a", { pos: [10, 10] });
  assert.deepEqual(rewiredCards(before, useDocStore.getState().graph), []);
});

test("Escape closes the inspector as well as clearing the selection", () => {
  const keys = readSrc("features/flows/editor/use-editor-shortcuts.ts");
  assert.match(keys, /ui\.setSelection\(\[\], \[\]\);\s*ui\.openInspector\(null\);/);
});

test("every fit stops at a zoom where card text still shows", async () => {
  const { FIT_MIN_ZOOM } = await import("../src/features/flows/editor/canvas-rules.ts");
  assert.ok(FIT_MIN_ZOOM > 0.45);
  for (const file of ["canvas.tsx", "canvas-controls.tsx", "toolbar.tsx"])
    assert.match(readSrc(`features/flows/editor/${file}`), /minZoom: FIT_MIN_ZOOM/, file);
});

test("a resized card writes lists and objects out in full", () => {
  const view = readSrc("features/flows/editor/flow-node.tsx");
  assert.match(view, /expanded=\{!!node\.size\}/);
  assert.match(view, /if \(expanded && typeof value === "object"\)/);
});
