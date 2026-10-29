// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A frame inside another draws above it, whatever order they were added in; a resize updates the size the router reads.

import assert from "node:assert/strict";
import test from "node:test";

import { doc } from "./flows-editor-fixtures.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore } = await import("../src/features/flows/stores/doc-store.ts");
const { RfDeriver, groupDepths } = await import("../src/features/flows/editor/derive-rf.ts");

const frame = (id: string, rect: [number, number, number, number]) => ({ id, label: id, color: null, rect });

test("an outer frame added after an inner one sits below it", () => {
  const d = doc([], []);
  d.groups = [frame("recipe", [100, 100, 400, 300]), frame("loop", [0, 0, 700, 600])];
  useDocStore.getState().load(d);
  const graph = useDocStore.getState().graph;
  assert.deepEqual([...groupDepths(graph)], [["recipe", 1], ["loop", 0]]);
  const nodes = new RfDeriver().nodes(graph, new Set(), null);
  const z = (id: string) => nodes.find((n) => n.id === `group:${id}`)?.zIndex ?? 0;
  assert.ok(z("recipe") > z("loop"), "inner above outer");
  assert.ok(z("recipe") < 0, "frames stay under the cards");
});

test("side-by-side frames share a level; identical ones stack in document order", () => {
  const d = doc([], []);
  d.groups = [frame("a", [0, 0, 200, 200]), frame("b", [300, 0, 200, 200]), frame("c", [0, 0, 200, 200])];
  useDocStore.getState().load(d);
  assert.deepEqual([...groupDepths(useDocStore.getState().graph)], [["a", 0], ["b", 0], ["c", 1]]);
});

test("a resize's sizes reach `measured`, so output wires follow the new right edge", () => {
  const src = readSrc("features/flows/editor/canvas.tsx");
  const handler = src.slice(src.indexOf("const onNodesChange"), src.indexOf("const onEdgesChange"));
  // The size update is its own branch, reached by resize changes too.
  assert.match(handler, /ended = true;\s*\}\s*\/\/[^\n]*\n[^\n]*\n\s*if \(change\.type === "dimensions" && change\.dimensions\)/);
});
