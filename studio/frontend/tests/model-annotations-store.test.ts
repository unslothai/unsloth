// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { create } from "zustand";
import type * as Store from "../src/features/hub/stores/model-annotations-store.ts";
import type * as Api from "../src/features/hub/api/model-annotations.ts";
import { readFastApiError } from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const DETAILS = {
  display_name: "My coder",
  labels: ["work"],
  notes: "Private note",
};

/** Use real Zustand, client and store code, replacing only HTTP and the login event boundary. */
function harness(
  fetch: (url: string, init?: RequestInit) => Promise<Response>,
) {
  let epoch = 0;
  const events = new EventTarget();
  const auth = {
    authFetch: fetch,
    getAuthSessionEpoch: () => epoch,
    AUTH_SESSION_CLEARED_EVENT: "test:cleared",
    AUTH_SESSION_STORED_EVENT: "test:stored",
  };
  const api = loadWithStubs<typeof Api>(
    new URL("../src/features/hub/api/model-annotations.ts", import.meta.url),
    {
      "@/features/auth": auth,
      "@/lib/format-fastapi-error": { readFastApiError },
    },
  );
  const priorWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: events,
  });
  let store: typeof Store;
  try {
    store = loadWithStubs<typeof Store>(
      new URL(
        "../src/features/hub/stores/model-annotations-store.ts",
        import.meta.url,
      ),
      {
        "@/features/auth": auth,
        zustand: { create },
        "../api/model-annotations": api,
      },
    );
  } finally {
    if (priorWindow) Object.defineProperty(globalThis, "window", priorWindow);
    else Reflect.deleteProperty(globalThis, "window");
  }
  return {
    ...store,
    changeAccount() {
      epoch += 1;
      events.dispatchEvent(new Event(auth.AUTH_SESSION_CLEARED_EVENT));
    },
  };
}

/** Resolve a controlled HTTP response after an account change or an overlapping call. */
function responseLater() {
  let finish!: (response: Response) => void;
  const promise = new Promise<Response>((resolve) => {
    finish = resolve;
  });
  return { promise, finish };
}

test("parallel mounts share one bulk read instead of fetching per model", async () => {
  let reads = 0;
  const pending = responseLater();
  const store = harness(async (url) => {
    reads += 1;
    assert.equal(url, "/api/settings/model-annotations");
    return pending.promise;
  });
  const first = store.loadModelAnnotations();
  assert.equal(store.loadModelAnnotations(), first);
  pending.finish(Response.json({ annotations: { model: DETAILS } }));
  await first;
  await store.loadModelAnnotations();
  assert.equal(reads, 1);
  assert.equal(store.useModelAnnotationsStore.getState().loaded, true);
  assert.deepEqual(
    store.useModelAnnotationsStore.getState().annotations.model,
    DETAILS,
  );
});

test("failed reads stay visible and can be retried without inventing an empty saved state", async () => {
  let reads = 0;
  const store = harness(async () =>
    ++reads === 1
      ? Response.json({ detail: "Offline" }, { status: 503 })
      : Response.json({ annotations: { model: DETAILS } }),
  );
  await store.loadModelAnnotations();
  assert.equal(store.useModelAnnotationsStore.getState().loaded, false);
  assert.match(
    store.useModelAnnotationsStore.getState().error ?? "",
    /Offline/,
  );
  await store.loadModelAnnotations();
  assert.equal(store.useModelAnnotationsStore.getState().error, null);
  assert.deepEqual(
    store.useModelAnnotationsStore.getState().annotations.model,
    DETAILS,
  );
});

test("a previous account's late read cannot replace the new account's notes or hydration", async () => {
  const old = responseLater();
  const next = responseLater();
  let reads = 0;
  const store = harness(async () =>
    ++reads === 1 ? old.promise : next.promise,
  );
  const first = store.loadModelAnnotations();
  store.changeAccount();
  const second = store.loadModelAnnotations();
  old.finish(Response.json({ annotations: { alice: DETAILS } }));
  await first;
  assert.deepEqual(store.useModelAnnotationsStore.getState().annotations, {});
  assert.equal(store.useModelAnnotationsStore.getState().loaded, false);
  assert.equal(store.loadModelAnnotations(), second);
  next.finish(
    Response.json({ annotations: { bob: { ...DETAILS, notes: "Bob" } } }),
  );
  await second;
  assert.deepEqual(
    Object.keys(store.useModelAnnotationsStore.getState().annotations),
    ["bob"],
  );
});

test("failed saves preserve saved data and keep the editor target open", async () => {
  const store = harness(async (_url, init) =>
    init?.method === "PUT"
      ? Response.json({ detail: "Cannot save" }, { status: 503 })
      : Response.json({ annotations: { model: DETAILS } }),
  );
  await store.loadModelAnnotations();
  store.editModelAnnotation({
    id: "model",
    title: "Original",
    annotation: DETAILS,
  });
  await assert.rejects(
    store.saveModelAnnotation("model", { ...DETAILS, notes: "New note" }),
    /Cannot save/,
  );
  assert.equal(store.useModelAnnotationsStore.getState().editing?.id, "model");
  assert.deepEqual(
    store.useModelAnnotationsStore.getState().annotations.model,
    DETAILS,
  );
});

test("a confirmed clear removes only its row and sends the original inventory id", async () => {
  const store = harness(async (_url, init) => {
    if (init?.method === "PUT") {
      assert.deepEqual(JSON.parse(String(init.body)), {
        model_id: "original-id",
        annotation: { display_name: "", labels: [], notes: "" },
      });
      return Response.json({ annotation: null });
    }
    return Response.json({
      annotations: { "original-id": DETAILS, other: DETAILS },
    });
  });
  await store.loadModelAnnotations();
  store.editModelAnnotation({
    id: "original-id",
    title: "Original",
    annotation: DETAILS,
  });
  await store.saveModelAnnotation("original-id", {
    display_name: "",
    labels: [],
    notes: "",
  });
  assert.deepEqual(
    Object.keys(store.useModelAnnotationsStore.getState().annotations),
    ["other"],
  );
  assert.equal(store.useModelAnnotationsStore.getState().editing, null);
});

test("a late save cannot restore old notes or close the next account's editor", async () => {
  const pending = responseLater();
  const store = harness(async (_url, init) =>
    init?.method === "PUT"
      ? pending.promise
      : Response.json({ annotations: {} }),
  );
  await store.loadModelAnnotations();
  const saving = store.saveModelAnnotation("model", DETAILS);
  store.changeAccount();
  await store.loadModelAnnotations();
  store.editModelAnnotation({ id: "model", title: "Bob's model" });
  pending.finish(Response.json({ annotation: DETAILS }));
  await saving;
  assert.deepEqual(store.useModelAnnotationsStore.getState().annotations, {});
  assert.equal(
    store.useModelAnnotationsStore.getState().editing?.title,
    "Bob's model",
  );
});
