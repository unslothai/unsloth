// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc, node } from "./flows-editor-fixtures.ts";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const api = await import("../src/features/flows/api.ts");
const { useDocStore, documentFromState, documentSignature } = await import(
  "../src/features/flows/stores/doc-store.ts"
);
const A = await import("../src/features/flows/editor/use-autosave.ts");

test("switching flows while a save is in flight still saves the last edit", async () => {
  // The server refuses a save whose baseUpdatedAt is not the stored one, as the real PUT does.
  const stored = { updatedAt: 1, doc: null as unknown };
  const saved: unknown[] = [];
  let release: () => void = () => undefined;
  const firstSave = new Promise<void>((r) => {
    release = r;
  });
  let puts = 0;
  api.setFlowsTransport(async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    puts += 1;
    if (puts === 1) await firstSave;
    if (body.baseUpdatedAt !== stored.updatedAt)
      return new Response(JSON.stringify({ detail: "conflict" }), {
        status: 409,
      });
    stored.updatedAt += 1;
    stored.doc = body.doc;
    saved.push(body.doc);
    return new Response(
      JSON.stringify({
        id: "a",
        name: "A",
        doc: body.doc,
        updatedAt: stored.updatedAt,
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  });
  const first = doc([node("x", "input.text", [0, 0], { value: "one" })]);
  first.id = "a";
  useDocStore.getState().load(first);
  A.beginSaveSession("a", 1, documentSignature(first));
  // An edit is being saved...
  useDocStore
    .getState()
    .load({
      ...first,
      nodes: [node("x", "input.text", [0, 0], { value: "two" })],
    });
  const flushing = A.flushSave();
  while (puts === 0) await new Promise((r) => setTimeout(r, 1));
  // ...another edit lands, then the user opens another flow before the first save returns.
  useDocStore
    .getState()
    .load({
      ...first,
      nodes: [node("x", "input.text", [0, 0], { value: "three" })],
    });
  A.endSaveSession();
  A.beginSaveSession("b", 1, "other");
  release();
  await flushing;
  for (let i = 0; i < 20 && saved.length < 2; i++)
    await new Promise((r) => setTimeout(r, 10));
  api.setFlowsTransport(null);
  const last = saved.at(-1) as ReturnType<typeof documentFromState>;
  assert.equal(last?.nodes[0]?.params.value, "three");
});
