// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, edge, node } from "./flows-editor-fixtures.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useDocStore, documentFromState, documentSignature } = await import(
  "../src/features/flows/stores/doc-store.ts"
);

test("the saved document round-trips through the store", () => {
  const original = doc(
    [
      node("a", "input.text", [1, 2], { value: "x" }),
      node("b", "output.preview"),
    ],
    [edge("e", "a", "text", "b", "value")],
  );
  original.groups = [{ id: "g", label: "G", rect: [0, 0, 10, 10] }];
  original.notes = [{ id: "n", text: "hi", rect: [0, 0, 5, 5] }];
  useDocStore.getState().load(original);
  assert.deepEqual(documentFromState(useDocStore.getState()), original);
});

test("edges to missing nodes are dropped on load", () => {
  useDocStore
    .getState()
    .load(
      doc(
        [node("a", "input.text")],
        [edge("e", "a", "text", "ghost", "value")],
      ),
    );
  assert.equal(useDocStore.getState().graph.edgeOrder.length, 0);
});

test("the signature ignores viewport state but sees every real edit", () => {
  const d = doc([node("a", "input.text")]);
  useDocStore.getState().load(d);
  const sig = documentSignature(documentFromState(useDocStore.getState()));
  assert.equal(
    documentSignature({ ...d, editor: { viewport: { x: 1, y: 2, zoom: 3 } } }),
    sig,
  );
  useDocStore.getState().moveNodes({ a: [5, 5] });
  const moved = documentSignature(documentFromState(useDocStore.getState()));
  assert.notEqual(moved, sig, "a committed move is saved");
  useDocStore.getState().setMeta({ name: "Renamed" });
  assert.notEqual(
    documentSignature(documentFromState(useDocStore.getState())),
    moved,
  );
});

test("only semantic changes bump the version autosave watches", () => {
  useDocStore.getState().load(doc([node("a", "input.text", [0, 0])]));
  const v = useDocStore.getState().version;
  useDocStore.getState().moveNodes({ a: [0, 0] });
  assert.equal(
    useDocStore.getState().version,
    v,
    "a no-op move is not a change",
  );
  useDocStore.getState().updateParams("missing", { x: 1 });
  assert.equal(useDocStore.getState().version, v);
});
