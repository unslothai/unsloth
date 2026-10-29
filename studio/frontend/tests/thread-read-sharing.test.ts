// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import * as abortSignals from "../src/features/hub/lib/abort-signals.ts";
import { skillLoadCardEvent } from "../src/features/chat/api/skill-load-event.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Row = { id: string; title: string };
type Module = {
  getChatThread: (
    threadId: string,
    options?: { bounded?: boolean; timeoutMs?: number; signal?: AbortSignal; share?: boolean },
  ) => Promise<Row | null>;
  updateChatThread: (threadId: string, patch: object, options?: { expectedTitle?: string; preserveTitleModifiedAt?: boolean }) => Promise<Row>;
  saveChatMessage: (message: { id: string; threadId: string }) => Promise<unknown>;
};

type Pending = {
  url: string;
  init?: RequestInit;
  resolve: (status: number, body: unknown) => void;
  reject: (error: unknown) => void;
};

function harness() {
  const pending: Pending[] = [];
  const auth = { epoch: 1 };
  const module = loadWithStubs<Module>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "./skill-load-event": { skillLoadCardEvent },
      "./thread-read-cache.ts": loadWithStubs(
        new URL("../src/features/chat/api/thread-read-cache.ts", import.meta.url),
        {},
      ),
      "@/features/auth": {
        getAuthSessionEpoch: () => auth.epoch,
        authFetch: (url: string, init?: RequestInit) =>
          new Promise((resolve, reject) => {
            const entry: Pending = {
              url,
              init,
              resolve: (status, body) =>
                resolve({
                  status,
                  ok: status >= 200 && status < 300,
                  headers: { get: () => null },
                  async json() {
                    return body;
                  },
                }),
              reject,
            };
            init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
            pending.push(entry);
          }),
      },
      "@/lib/format-fastapi-error": { formatApiErrorBody: () => null },
      "../types": {},
      "../types/api": {},
      "../utils/chat-history-revision": {
        publishChatHistoryRevision: () => {},
        CHAT_HISTORY_REVISION_KEY: "unsloth_chat_history_revision",
      },
      "../utils/openapi-support": { readAdvertisedPatchGuards: () => null },
      "../utils/load-warning-toast": { showLoadWarning: () => {} },
      "../lib/load-request": {},
      "./generation-length.ts": {},
      "./gguf-variants-request": {},
      "./padded-response": { assertCompletedPaddedBody: () => {} },
      "@/features/hf-auth": { prepareHfTokenForUse: async () => undefined },
      "@/features/settings/low-disk-check": { checkDiskSpace: () => Promise.resolve() },
      "@/features/igpu-carveout": {
        dismissCarveoutAdviceForModel: () => {},
        showCarveoutAdvice: () => {},
      },
      "@/features/hub/lib/abort-signals": abortSignals,
      "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
      "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
      "@/features/native-intents/api": { consumeNativePathToken: () => undefined },
      "@/lib/model-lifecycle-events": {},
    },
  );
  const requests = () => pending.filter((p) => p.url.endsWith("/api/chat/threads/t1") && !p.init?.method);
  return { module, pending, requests, auth };
}

const row: Row = { id: "t1", title: "T" };
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));

test("concurrent shared reads of one thread make one request and share its record", async () => {
  const { module, requests } = harness();
  const reads = Array.from({ length: 6 }, () => module.getChatThread("t1", { share: true }));
  assert.equal(requests().length, 1);
  requests()[0].resolve(200, row);
  const results = await Promise.all(reads);
  assert.ok(results.every((r) => r === results[0]));
  assert.deepEqual(results[0], row);
});

test("a read is only shared when asked: the default stays a private request", async () => {
  const { module, requests } = harness();
  void module.getChatThread("t1", { share: true });
  void module.getChatThread("t1");
  void module.getChatThread("t1", { bounded: true });
  assert.equal(requests().length, 3);
  for (const r of requests()) r.resolve(200, row);
  await tick();
});

test("there is no TTL: a shared read after the request settled starts a new one", async () => {
  const { module, requests } = harness();
  const first = module.getChatThread("t1", { share: true });
  requests()[0].resolve(200, row);
  await first;
  await tick();
  void module.getChatThread("t1", { share: true });
  assert.equal(requests().length, 2);
  requests()[1].resolve(200, row);
  await tick();
});

test("a read still pending past the join window is not joined, so one stalled socket cannot hold every later reader", async () => {
  const { module, requests } = harness();
  const real = performance.now.bind(performance);
  const stalled = module.getChatThread("t1", { share: true });
  try {
    performance.now = () => real() + 2_500;
    const later = module.getChatThread("t1", { share: true });
    assert.equal(requests().length, 2);
    const joined = module.getChatThread("t1", { share: true });
    assert.equal(requests().length, 2, "readers inside the new window join the fresh request");
    requests()[1].resolve(200, { ...row, title: "fresh" });
    assert.equal((await later)?.title, "fresh");
    assert.equal((await joined)?.title, "fresh");
  } finally {
    delete (performance as { now?: unknown }).now;
  }
  requests()[0].resolve(200, row);
  assert.deepEqual(await stalled, row);
});

test("a 404 is shared as null and a failure is shared as the same error", async () => {
  const { module, requests } = harness();
  const missing = [module.getChatThread("t1", { share: true }), module.getChatThread("t1", { share: true })];
  requests()[0].resolve(404, null);
  assert.deepEqual(await Promise.all(missing), [null, null]);
  await tick();
  const failing = [module.getChatThread("t1", { share: true }), module.getChatThread("t1", { share: true })];
  requests()[1].resolve(500, { detail: "boom" });
  const errors = await Promise.all(failing.map((p) => p.then(() => null, (e: unknown) => e)));
  assert.ok(errors[0] instanceof Error);
  assert.equal(errors[0], errors[1]);
});

test("one reader's deadline detaches that reader alone, and a retry starts a fresh request", async () => {
  const { module, requests } = harness();
  const quiet = module.getChatThread("t1", { share: true });
  const bounded = module.getChatThread("t1", { share: true, timeoutMs: 1 });
  // AbortSignal.timeout's timer does not hold the event loop open (seen on Node 22 CI).
  const alive = setTimeout(() => {}, 10_000);
  await assert.rejects(bounded, (e: DOMException) => e.name === "TimeoutError");
  clearTimeout(alive);
  assert.equal(requests()[0].init?.signal?.aborted, false);
  const retry = module.getChatThread("t1", { share: true, timeoutMs: 1000 });
  assert.equal(requests().length, 2);
  requests()[1].resolve(200, { ...row, title: "fresh" });
  assert.equal((await retry)?.title, "fresh");
  requests()[0].resolve(200, row);
  assert.deepEqual(await quiet, row);
});

test("the request itself is aborted only when every reader with a signal has gone", async () => {
  const { module, requests } = harness();
  const a = new AbortController();
  const b = new AbortController();
  const ra = module.getChatThread("t1", { share: true, signal: a.signal });
  const rb = module.getChatThread("t1", { share: true, signal: b.signal });
  a.abort();
  await assert.rejects(ra, (e: DOMException) => e.name === "AbortError");
  assert.equal(requests()[0].init?.signal?.aborted, false);
  b.abort();
  await assert.rejects(rb, (e: DOMException) => e.name === "AbortError");
  assert.equal(requests()[0].init?.signal?.aborted, true);
});

test("a thread write settling drops that thread's in-flight read, so a later reader starts fresh", async () => {
  const { module, pending, requests } = harness();
  const older = module.getChatThread("t1", { share: true });
  const patch = module.updateChatThread("t1", { title: "renamed" });
  await tick();
  const patchRequest = pending.find((p) => p.init?.method === "PATCH")!;
  patchRequest.resolve(200, { ...row, title: "renamed" });
  await patch;
  void module.getChatThread("t1", { share: true });
  assert.equal(requests().length, 2);
  requests()[0].resolve(200, row);
  requests()[1].resolve(200, { ...row, title: "renamed" });
  assert.equal((await older)?.title, "T");
  await tick();
});

test("a message save in another thread leaves this thread's shared read joinable", async () => {
  const { module, pending, requests } = harness();
  void module.getChatThread("t1", { share: true });
  const save = module.saveChatMessage({ id: "m1", threadId: "other" });
  await tick();
  pending.find((p) => p.init?.method === "PUT")!.resolve(200, { id: "m1", threadId: "other" });
  await save;
  void module.getChatThread("t1", { share: true });
  assert.equal(requests().length, 1);
  requests()[0].resolve(200, row);
  await tick();
});

test("a read started under one auth session is never joined by the next", async () => {
  const { module, requests, auth } = harness();
  const before = module.getChatThread("t1", { share: true });
  auth.epoch += 1;
  const after = module.getChatThread("t1", { share: true });
  await tick();
  assert.equal(requests().length, 2);
  requests()[0].resolve(200, { id: "t1", title: "first session" });
  requests()[1].resolve(200, row);
  assert.deepEqual(await before, { id: "t1", title: "first session" });
  assert.deepEqual(await after, row);
});

test("title regeneration sends the recency option while an ordinary rename omits it", async () => {
  for (const regenerate of [false, true]) {
    const { module, pending } = harness();
    const write = module.updateChatThread("t1", { title: "New title" },
      regenerate ? { expectedTitle: "T", preserveTitleModifiedAt: true } : undefined);
    await tick();
    const request = pending.find((p) => p.init?.method === "PATCH")!;
    assert.deepEqual(JSON.parse(String(request.init?.body)), regenerate
      ? { title: "New title", expectedTitle: "T", preserveTitleModifiedAt: true }
      : { title: "New title" });
    request.resolve(200, { ...row, title: "New title" });
    assert.equal((await write).title, "New title");
  }
});
