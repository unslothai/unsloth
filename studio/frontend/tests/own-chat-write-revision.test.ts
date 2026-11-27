// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import * as abortSignals from "../src/features/hub/lib/abort-signals.ts";
import { skillLoadCardEvent } from "../src/features/chat/api/skill-load-event.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type OwnWrite = {
  threadId: string;
  thread: readonly [number, number];
  history: readonly [number, number];
};
type Module = {
  updateChatThread: (threadId: string, patch: object) => Promise<unknown>;
  saveChatMessage: (message: {
    id: string;
    threadId: string;
  }) => Promise<unknown>;
  syncChatMessages: (threadId: string, messages: unknown[]) => Promise<unknown>;
  subscribeOwnChatHistoryWrites: (
    listener: (write: OwnWrite) => void,
  ) => () => void;
};

type Pending = {
  url: string;
  init?: RequestInit;
  resolve: (status: number, body: unknown) => void;
  reject: (error: unknown) => void;
};

function harness(header: string | null) {
  const pending: Pending[] = [];
  const auth = { epoch: 1 };
  const module = loadWithStubs<Module>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "./skill-load-event": { skillLoadCardEvent },
      "./thread-read-cache.ts": loadWithStubs(
        new URL(
          "../src/features/chat/api/thread-read-cache.ts",
          import.meta.url,
        ),
        {},
      ),
      "@/lib/sse-framing": loadWithStubs(
        new URL("../src/lib/sse-framing.ts", import.meta.url),
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
                  headers: {
                    get: (name: string) =>
                      name.toLowerCase() === "x-unsloth-chat-revision"
                        ? header
                        : null,
                  },
                  async json() {
                    return body;
                  },
                }),
              reject,
            };
            init?.signal?.addEventListener(
              "abort",
              () => reject(init.signal?.reason),
              { once: true },
            );
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
      "@/features/settings/low-disk-check": {
        checkDiskSpace: () => Promise.resolve(),
      },
      "@/features/igpu-carveout": {
        dismissCarveoutAdviceForModel: () => {},
        showCarveoutAdvice: () => {},
      },
      "@/features/hub/lib/abort-signals": abortSignals,
      "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
      "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
      "@/features/native-intents/api": {
        consumeNativePathToken: () => undefined,
      },
      "@/lib/model-lifecycle-events": {},
    },
  );
  return { module, pending };
}

const tick = () => new Promise<void>((resolve) => setImmediate(resolve));

test("a chat write reports the revisions its transaction moved", async () => {
  const { module, pending } = harness("4,6,40,42");
  const seen: OwnWrite[] = [];
  const unsubscribe = module.subscribeOwnChatHistoryWrites((write) =>
    seen.push(write),
  );
  const saved = module.saveChatMessage({ id: "m1", threadId: "t1" });
  await tick();
  pending[0].resolve(200, { id: "m1", threadId: "t1" });
  await saved;
  const synced = module.syncChatMessages("t1", []);
  await tick();
  pending[1].resolve(200, { messages: [] });
  await synced;
  assert.deepEqual(seen, [
    { threadId: "t1", thread: [4, 6], history: [40, 42] },
    { threadId: "t1", thread: [4, 6], history: [40, 42] },
  ]);
  unsubscribe();
  const patched = module.updateChatThread("t1", { title: "x" });
  await tick();
  pending[2].resolve(200, { id: "t1" });
  await patched;
  assert.equal(seen.length, 2, "an unsubscribed listener hears nothing");
});

for (const [name, header, status] of [
  ["an older server sends no header", null, 200],
  ["a malformed header", "4,6,x,42", 200],
  ["a failed write", "4,6,40,42", 409],
] as const) {
  test(`${name}: nothing is reported, so the poll re-reads as before`, async () => {
    const { module, pending } = harness(header);
    const seen: OwnWrite[] = [];
    module.subscribeOwnChatHistoryWrites((write) => seen.push(write));
    const patched = module.updateChatThread("t1", { title: "x" });
    await tick();
    pending[0].resolve(
      status,
      status === 200 ? { id: "t1" } : { detail: "changed" },
    );
    await patched.catch(() => undefined);
    assert.deepEqual(seen, []);
  });
}
