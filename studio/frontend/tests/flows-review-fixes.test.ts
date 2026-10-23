// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, node } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

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
