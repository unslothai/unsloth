// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

register("./helpers/toast-resolver.mjs", import.meta.url);

const { calls } = await import("./helpers/toast-stub.mjs");
const { LOAD_WARNING_TOAST_ID, showLoadWarning } = await import(
  "../src/features/chat/utils/load-warning-toast.ts"
);

const NOTICE =
  "Not enough disk space to download BF16 (7.5 GB needed, 7.5 GB free), so Q4_1 (2.4 GB) was loaded instead.";
const FROM_STATUS_ON_MODEL_CHANGE =
  /if \(hydratingExistingModel\) \{\s*showLoadWarning\(status\.memory_warning\);/;

type ChatApi = {
  loadModel: (payload: Record<string, unknown>) => Promise<Record<string, unknown>>;
};

function chatApi(body: Record<string, unknown>) {
  const shown: (string | null | undefined)[] = [];
  const module = loadWithStubs<ChatApi>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async () => ({
          status: 200,
          ok: true,
          headers: { get: () => null },
          json: async () => body,
        }),
      },
      "@/lib/format-fastapi-error": { formatApiErrorBody: () => null },
      "../types": {},
      "../types/api": {},
      "../utils/chat-history-revision": {
        notifyChatHistoryUpdated: () => {},
        isCoalescedHistoryEvent: () => false,
      },
      "../utils/load-warning-toast": {
        showLoadWarning: (warning: string | null | undefined) => shown.push(warning),
      },
      "./generation-length.ts": {},
      "./gguf-variants-request": {},
      "./padded-response": { assertCompletedPaddedBody: () => {} },
      "@/features/hf-auth": { prepareHfTokenForUse: async () => ({ proceed: true }) },
      "@/features/igpu-carveout": {
        dismissCarveoutAdviceForModel: () => {},
        showCarveoutAdvice: () => {},
      },
      "@/features/hub/lib/abort-signals": {},
      "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
      "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
      "@/features/native-intents/api": { consumeNativePathToken: () => undefined },
      "@/lib/model-lifecycle-events": {
        withModelLoadNotice: async (
          _runtime: string,
          _path: string | null,
          run: () => Promise<unknown>,
        ) => run(),
      },
    },
  );
  return { module, shown };
}

test("a load warning raises one warning toast carrying the backend's text", () => {
  calls.length = 0;
  showLoadWarning(NOTICE);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].kind, "warning");
  assert.equal(calls[0].options?.id, LOAD_WARNING_TOAST_ID);
  assert.equal(calls[0].options?.description, NOTICE);
});

test("a model with no warning takes the previous model's warning down", () => {
  calls.length = 0;
  showLoadWarning(null);
  showLoadWarning(undefined);
  assert.deepEqual(calls, [
    { kind: "dismiss", id: LOAD_WARNING_TOAST_ID },
    { kind: "dismiss", id: LOAD_WARNING_TOAST_ID },
  ]);
});

test("a load response's warning reaches the notice, and its absence takes one down", async () => {
  const warned = chatApi({ status: "loaded", model: "m", memory_warning: NOTICE });
  await warned.module.loadModel({ model_path: "m" });
  assert.deepEqual(warned.shown, [NOTICE]);

  const quiet = chatApi({ status: "loaded", model: "m" });
  await quiet.module.loadModel({ model_path: "m" });
  assert.deepEqual(quiet.shown, [undefined]);
});

// No test drives applyActiveModelStatusToStore, so a model loaded elsewhere is matched here.
test("a model loaded elsewhere reaches the toast on the model change", () => {
  assert.match(
    readSrc("features/chat/lib/apply-inference-status-to-store.ts"),
    FROM_STATUS_ON_MODEL_CHANGE,
  );
});
