// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, node } from "./flows-editor-fixtures.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { cloneFragment, useDocStore } = await import(
  "../src/features/flows/stores/doc-store.ts"
);

const { isStudioMediaUrl } = await import("../src/features/flows/media-url.ts");

test("only Studio's own media paths are fetched with the user's token", () => {
  assert.equal(isStudioMediaUrl("/api/inference/images/gallery/1/file"), true);
  assert.equal(isStudioMediaUrl("https://evil.example/x.png"), false);
  assert.equal(isStudioMediaUrl("//evil.example/x.png"), false);
  assert.equal(isStudioMediaUrl("/api//evil.example"), false);
  assert.equal(isStudioMediaUrl(null), false);
});

test("deleting a group frame releases the nodes it held", () => {
  const d = doc([
    { ...node("fe", "logic.for_each"), group: "g1" },
    { ...node("t", "text.template", [300, 0]), group: "g1" },
  ]);
  d.groups = [{ id: "g1", label: "Loop", rect: [0, 0, 600, 300] }];
  useDocStore.getState().load(d);
  useDocStore.getState().removeElements({ groups: ["g1"] });
  const nodes = useDocStore.getState().graph.nodes;
  assert.equal(nodes.fe.group ?? null, null);
  assert.equal(nodes.t.group ?? null, null);
});

test("a copied node does not join the original's For Each body", () => {
  const d = doc([
    { ...node("fe", "logic.for_each"), group: "g1" },
    { ...node("t", "text.template", [300, 0]), group: "g1" },
  ]);
  d.groups = [{ id: "g1", label: "Loop", rect: [0, 0, 600, 300] }];
  useDocStore.getState().load(d);
  const fragment = cloneFragment(
    useDocStore.getState().graph,
    ["fe", "t"],
    [900, 0],
  );
  assert.equal(fragment.nodes.length, 2);
  for (const copy of fragment.nodes) assert.equal(copy.group, null);
  assert.equal(useDocStore.getState().graph.nodes.t.group, "g1");
});

test("a pasted copy joins the frame it lands in, and only that one", () => {
  const d = doc([
    { ...node("fe", "logic.for_each"), group: "g1" },
    { ...node("t", "text.template", [100, 100]), group: "g1" },
  ]);
  d.groups = [{ id: "g1", label: "Loop", rect: [0, 0, 800, 400] }];
  useDocStore.getState().load(d);
  const store = useDocStore.getState();
  const inside = cloneFragment(store.graph, ["t"], [40, 40]);
  const outside = cloneFragment(store.graph, ["t"], [2000, 0]);
  store.insertFragment(inside);
  useDocStore.getState().insertFragment(outside);
  const nodes = useDocStore.getState().graph.nodes;
  assert.equal(nodes[inside.nodes[0].id].group, "g1");
  assert.equal(nodes[outside.nodes[0].id].group, null);
});

test("resizing a frame re-parents the nodes it takes in or leaves out", () => {
  const d = doc([
    { ...node("fe", "logic.for_each"), group: "g1" },
    { ...node("in", "text.template", [100, 100]), group: "g1" },
    node("out", "text.template", [900, 100]),
  ]);
  d.groups = [{ id: "g1", label: "Loop", rect: [0, 0, 600, 300] }];
  useDocStore.getState().load(d);
  const g = useDocStore.getState().graph.groups.g1;
  useDocStore.getState().resizeGroup({ ...g, rect: [0, 0, 1300, 300] });
  assert.equal(useDocStore.getState().graph.nodes.out.group, "g1");
  useDocStore.getState().resizeGroup({ ...g, rect: [0, 0, 380, 80] });
  const nodes = useDocStore.getState().graph.nodes;
  assert.equal(nodes.fe.group, "g1");
  assert.equal(nodes.in.group ?? null, null);
  assert.equal(nodes.out.group ?? null, null);
});

test("a block from a newer revision opens the flow read-only", async () => {
  const { needsNewerStudio, useCatalogStore } = await import(
    "../src/features/flows/stores/catalog-store.ts"
  );
  useCatalogStore.setState({
    specs: { "text.template": { type: "text.template", v: 1 } as never },
  });
  assert.equal(needsNewerStudio({ type: "text.template", v: 1 }), false);
  assert.equal(needsNewerStudio({ type: "text.template" }), false);
  assert.equal(needsNewerStudio({ type: "text.template", v: 2 }), true);
  assert.equal(needsNewerStudio({ type: "made.up" }), true);
});

test("a run snapshot's progress object becomes a fraction", async () => {
  const { progressFraction } = await import(
    "../src/features/flows/stores/runtime-store.ts"
  );
  assert.equal(progressFraction({ fraction: 0.25, phase: "x" }), 0.25);
  assert.equal(progressFraction({ step: 3, total: 4 }), 0.75);
  assert.equal(progressFraction({ phase: "loading" }), null);
  assert.equal(progressFraction(null), null);
  assert.equal(progressFraction(0.5), 0.5);
});

test("a node added inside a frame joins it", () => {
  const d = doc([{ ...node("fe", "logic.for_each"), group: "g1" }]);
  d.groups = [{ id: "g1", label: "Loop", rect: [0, 0, 600, 300] }];
  useDocStore.getState().load(d);
  useDocStore.getState().addNode(node("inside", "text.template", [100, 100]));
  useDocStore.getState().addNode(node("outside", "text.template", [900, 100]));
  const nodes = useDocStore.getState().graph.nodes;
  assert.equal(nodes.inside.group, "g1");
  assert.equal(nodes.outside.group ?? null, null);
});

test("dialogue written while the reader is mid-scene is kept for the next visit", () => {
  const src = readSrc("features/flows/player/story/story-player.tsx");
  assert.match(src, /heldLines\.current\.set\(key, written\.lines\)/);
  const visit = src.indexOf("const held = heldLines.current.get(key);");
  assert.ok(visit > 0 && visit < src.indexOf("!requested.current.has(key)"));
});

test("leaving the editor waits for a successful save, or an explicit Leave anyway", () => {
  const src = readSrc("features/flows/editor/flows-editor-page.tsx");
  const back = src.slice(src.indexOf("onBack={"), src.indexOf("onPlay={"));
  assert.match(back, /leaveAfterSave\(/);
  const leave = src.slice(src.indexOf("function leaveAfterSave"), src.indexOf("type LoadState"));
  assert.match(leave, /flushSave\(\)[\s\S]*?if \(saved === true\) return go\(\);/);
  assert.match(leave, /label: "Leave anyway", onClick: go/);
  assert.ok(!leave.includes(".finally("));
});

test("a second Run click while the first start request is pending is ignored", () => {
  const src = readSrc("features/flows/editor/run-controller.ts");
  const body = src.slice(src.indexOf("export async function runFlow"));
  assert.ok(body.indexOf("starting = true") < body.indexOf("await startRun("));
  assert.match(body.slice(0, body.indexOf("starting = true")), /if \(\s*starting \|\|/);
  assert.match(body, /finally \{\s*starting = false;/);
});

test("the inspector shows a Preview block's value, not an empty Output section", () => {
  const src = readSrc("features/flows/editor/inspector.tsx");
  const outputs = src.slice(src.indexOf("function NodeOutputs"), src.indexOf("function ParamRow"));
  assert.match(outputs, /spec\.outputs\.length === 0 &&[\s\S]*?Object\.entries\(runtime\.outputs/);
});
