// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, node } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore } = await import(
  "../src/features/flows/stores/doc-store.ts"
);
const { isStudioMediaUrl } = await import(
  "../src/features/flows/media-url.ts"
);

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
