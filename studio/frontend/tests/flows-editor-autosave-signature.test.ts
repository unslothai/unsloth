// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, edge, node } from "./helpers/flows.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const api = await import("../src/features/flows/api.ts");
const { useDocStore, documentFromState, documentSignature } = await import(
  "../src/features/flows/stores/doc-store.ts"
);
const A = await import("../src/features/flows/editor/use-autosave.ts");

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

/** A promise plus its resolver, so the test decides when each step happens. */
function signal(): { promise: Promise<void>; resolve: () => void } {
  let resolve: () => void = () => undefined;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

// The timeout only bounds a regression that never sends the closing save; it never paces the test.
test("switching flows while a save is in flight still saves the last edit", { timeout: 10_000 }, async () => {
  const stored = { updatedAt: 1 };
  const saved: ReturnType<typeof documentFromState>[] = [];
  const firstSave = signal();
  const firstPut = signal();
  const secondSaved = signal();
  let puts = 0;
  api.setFlowsTransport(async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    puts += 1;
    if (puts === 1) {
      firstPut.resolve();
      await firstSave.promise;
    }
    if (body.baseUpdatedAt !== stored.updatedAt)
      return new Response(JSON.stringify({ detail: "conflict" }), { status: 409 });
    stored.updatedAt += 1;
    saved.push(body.doc);
    if (saved.length === 2) secondSaved.resolve();
    return new Response(
      JSON.stringify({ id: "a", name: "A", doc: body.doc, updatedAt: stored.updatedAt }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  });
  const withValue = (value: string) => ({
    ...first,
    nodes: [node("x", "input.text", [0, 0], { value })],
  });
  const first = doc([node("x", "input.text", [0, 0], { value: "one" })]);
  first.id = "a";
  useDocStore.getState().load(first);
  A.beginSaveSession("a", 1, documentSignature(first));
  useDocStore.getState().load(withValue("two"));
  const flushing = A.flushSave();
  await firstPut.promise;
  useDocStore.getState().load(withValue("three"));
  A.endSaveSession();
  A.beginSaveSession("b", 1, "other");
  firstSave.resolve();
  await flushing;
  await secondSaved.promise;
  api.setFlowsTransport(null);
  assert.deepEqual(saved.map((d) => d.nodes[0]?.params.value), ["two", "three"]);
});
