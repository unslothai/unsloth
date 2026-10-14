// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test, { afterEach } from "node:test";

import { AUTH_SESSION_CLEARED_EVENT } from "../src/features/auth/session-events.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/igpu-carveout-resolver.mjs", import.meta.url);
const { fireWindowEvent } = installLocalStorageFake();

const { calls } = await import("./helpers/toast-stub.mjs");
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { toastEmbeddingModelSaved, useReembedStore } = await import(
  "../src/features/rag/components/reembed-store.ts"
);

const job = (status: string, failed = 0, id = "j1") => ({
  id,
  status,
  done: 2,
  failed,
  remaining: 0,
  clippedChunks: 0,
  progress: 1,
  error: failed ? "embedder down" : null,
});

const idle = { staleDocuments: 0, staleArchived: 0 };
const stale = { staleDocuments: 2, staleArchived: 0 };
const kinds = () => calls.map((call) => call.kind);
const flush = () => new Promise((resolve) => setImmediate(resolve));
const offline = () => Promise.reject(new Error("offline"));
const reply = (body: object) => new Response(JSON.stringify(body));
const signOut = () => fireWindowEvent(AUTH_SESSION_CLEARED_EVENT, {});
afterEach(signOut);

function serve(status: object, posted: string[] = []) {
  calls.length = 0;
  setAuthFetchHandler((url, init) => {
    if (init?.method !== "POST") return reply(status);
    posted.push(url);
    return reply({ job: job("running") });
  });
  return posted;
}

test("signing out forgets the job and stops polling for it", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  serve({ ...stale, job: job("running", 0, "j11") });
  await useReembedStore.getState().load();
  setAuthFetchHandler(offline);
  await useReembedStore.getState().cancel(); // leaves the state to reconcile
  serve({ ...stale, job: job("running", 0, "j11") });
  void useReembedStore.getState().load(); // answered after signing out
  setAuthFetchHandler(offline);
  void useReembedStore.getState().start(); // fails after signing out
  signOut();
  await useReembedStore.getState().load(); // a read that fails once signed out
  serve({ ...idle, job: job("completed", 0, "j11") });
  t.mock.timers.tick(1500);
  await flush();
  assert.deepEqual([useReembedStore.getState().status, kinds()], [null, []]);
});

test("a save that leaves documents stale offers Re-embed, which starts the job", async () => {
  const posted = serve({ staleDocuments: 3, staleArchived: 0, job: null });
  await toastEmbeddingModelSaved("Saved");
  const [saved] = calls;
  assert.equal(
    saved.options?.description,
    "3 documents were indexed with a different embedding model, so semantic search skips them.",
  );
  assert.equal(saved.options?.duration, 10_000);
  saved.options?.action?.onClick();
  assert.deepEqual(posted, ["/api/rag/reembed"]);
});

test("no action when nothing is stale or a running job already follows the model", async () => {
  for (const status of [
    { ...idle, job: null },
    { ...stale, job: job("running") },
  ]) {
    serve(status);
    await toastEmbeddingModelSaved("Saved");
    assert.equal(calls[0].options?.action, undefined);
  }
});

test("a job's end is announced once, and only by a client that saw it running", async () => {
  serve({ ...idle, job: job("completed", 0, "j0") });
  await useReembedStore.getState().load();
  assert.equal(calls.length, 0);
  for (const [end, kind] of [
    [job("completed"), "success"],
    [job("completed", 1), "warning"],
    [job("failed", 3), "error"],
  ] as const) {
    serve({ ...idle, job: job("running") });
    await useReembedStore.getState().load();
    serve({ ...idle, job: end });
    await useReembedStore.getState().load();
    await useReembedStore.getState().load();
    assert.deepEqual(kinds(), [kind]);
  }
});

test("polling outlives failed requests, and a failed cancel reports false", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  serve({ ...idle, job: job("running", 0, "j2") });
  await useReembedStore.getState().load();
  setAuthFetchHandler(offline);
  assert.equal(await useReembedStore.getState().cancel(), false);
  t.mock.timers.tick(1500);
  await flush();
  serve({ ...idle, job: job("completed", 0, "j2") });
  t.mock.timers.tick(1500);
  await flush();
  assert.deepEqual(kinds(), ["success"]);
});

test("a status reply older than a start reply is replaced by a fresh read", async () => {
  let server: object = { ...idle, job: null };
  const held: { get?: (response: Response) => void } = {};
  calls.length = 0;
  setAuthFetchHandler((_url, init) => {
    if (init?.method === "POST") {
      server = { ...stale, job: job("running", 0, "j3") };
      return reply({ job: job("running", 0, "j3") });
    }
    if (held.get) return reply(server);
    return new Promise((resolve) => {
      held.get = resolve;
    });
  });
  const late = toastEmbeddingModelSaved("Saved");
  await useReembedStore.getState().start();
  held.get?.(reply({ ...stale, job: null }));
  await late;
  assert.equal(useReembedStore.getState().status?.job?.id, "j3");
  const saved = calls.at(-1)?.options;
  assert.equal(saved?.description?.startsWith("2 documents"), true);
  assert.equal(saved?.action, undefined);
});

test("a late start reply neither hides its job nor announces its end twice", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let server: object | null = { ...idle, job: null };
  const held: { post?: (response: Response) => void } = {};
  calls.length = 0;
  setAuthFetchHandler((_url, init) => {
    if (init?.method === "POST") {
      return new Promise((resolve) => {
        held.post = resolve;
      });
    }
    if (server === null) throw new Error("offline");
    return reply(server);
  });
  const startReply = (id: string) => reply({ job: job("running", 0, id) });
  let started = useReembedStore.getState().start();
  await useReembedStore.getState().load();
  server = null; // the read that replaces the dropped start reply fails once
  held.post?.(startReply("j5"));
  await started;
  await flush();
  server = { ...idle, job: job("running", 0, "j5") };
  t.mock.timers.tick(1500);
  await flush();
  assert.equal(useReembedStore.getState().status?.job?.id, "j5");
  server = { ...idle, job: job("completed", 0, "j5") };
  started = useReembedStore.getState().start();
  await useReembedStore.getState().load();
  held.post?.(startReply("j5"));
  await started;
  await flush();
  t.mock.timers.tick(1500);
  await flush();
  assert.deepEqual(kinds(), ["success"]);
});

test("a finished job in a start or cancel reply refreshes the stale counts", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  serve({ ...stale, job: job("running", 0, "j8") });
  await useReembedStore.getState().load();
  let online = false;
  setAuthFetchHandler((_url, init) => {
    if (init?.method === "POST") {
      return reply({ job: job("completed", 0, "j8") });
    }
    if (!online) throw new Error("offline");
    return reply({ ...idle, job: job("completed", 0, "j8") });
  });
  await useReembedStore.getState().cancel();
  await flush();
  online = true;
  t.mock.timers.tick(1500);
  await flush();
  assert.equal(useReembedStore.getState().status?.staleDocuments, 0);
});

test("a start whose reply was lost is found by the next status read", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  serve({ ...idle, job: null });
  await useReembedStore.getState().load();
  setAuthFetchHandler((_url, init) => {
    if (init?.method === "POST") throw new Error("offline");
    return reply({ ...idle, job: job("running", 0, "j9") });
  });
  await useReembedStore.getState().start();
  t.mock.timers.tick(1500);
  await flush();
  assert.equal(useReembedStore.getState().status?.job?.id, "j9");
});

test("a status reply sent before a failed start cannot end the recovery", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const held: { get?: (response: Response) => void } = {};
  setAuthFetchHandler((_url, init) => {
    if (init?.method === "POST") throw new Error("offline");
    if (held.get) {
      return reply({ ...idle, job: job("running", 0, "j10") });
    }
    return new Promise((resolve) => {
      held.get = resolve;
    });
  });
  const early = useReembedStore.getState().load();
  await useReembedStore.getState().start();
  held.get?.(reply({ ...idle, job: null }));
  await early;
  assert.equal(useReembedStore.getState().status?.job?.id, "j10");
});
