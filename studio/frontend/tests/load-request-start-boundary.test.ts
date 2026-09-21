// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** When `loadModel` announces that the load request is actually going out.
 *
 * The chat hook decides which log a failure toast offers from this, because only a request
 * that was SENT can have left a runner log behind. Everything `loadModel` does before it --
 * preparing the HF token, which can prompt and be cancelled, and the abort check -- reaches
 * the hook's catch with no runner started, and naming `llama-server` there opens whatever
 * unrelated earlier attempt is newest on the host.
 *
 * So the ordering inside `loadModel` is the contract, not an implementation detail: drive
 * the real module and assert the callback fires after a proceeding token preparation and
 * never without one.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Prepared = { proceed: boolean; token?: string | null };

type ChatApi = {
  loadModel: (
    payload: Record<string, unknown>,
    options?: { signal?: AbortSignal; onRequestStart?: () => void },
  ) => Promise<Record<string, unknown>>;
};

function chatApi(prepared: Prepared) {
  const order: string[] = [];
  const module = loadWithStubs<ChatApi>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async () => {
          order.push("request");
          return {
            status: 200,
            ok: true,
            headers: { get: () => null },
            json: async () => ({ status: "loaded", model: "m" }),
          };
        },
      },
      "@/lib/format-fastapi-error": { formatApiErrorBody: () => null },
      "../types": {},
      "../types/api": {},
      "../utils/chat-history-revision": {
        notifyChatHistoryUpdated: () => {},
        isCoalescedHistoryEvent: () => false,
      },
      "../utils/load-warning-toast": { showLoadWarning: () => {} },
      "./generation-length.ts": {},
      "./gguf-variants-request": {},
      "./padded-response": { assertCompletedPaddedBody: () => {} },
      "@/features/hf-auth": {
        prepareHfTokenForUse: async () => {
          order.push("token");
          return prepared;
        },
      },
      "@/features/igpu-carveout": {
        dismissCarveoutAdviceForModel: () => {},
        showCarveoutAdvice: () => {},
      },
      "@/features/hub/lib/abort-signals": {},
      "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
      "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
      "@/features/native-intents/api": {
        consumeNativePathToken: () => undefined,
      },
      "@/lib/model-lifecycle-events": {
        withModelLoadNotice: async (
          _runtime: string,
          _path: string | null,
          run: () => Promise<unknown>,
        ) => run(),
      },
    },
  );
  return { module, order };
}

test("the request-start callback fires after the token is prepared and before the POST", async () => {
  const { module, order } = chatApi({ proceed: true, token: null });
  await module.loadModel(
    { model_path: "m" },
    { onRequestStart: () => order.push("announced") },
  );
  assert.deepEqual(order, ["token", "announced", "request"]);
});

test("a cancelled token prompt never announces a request", async () => {
  // The case the hook cares about: a local or native GGUF pick skips the outer token
  // preparation, so an invalid stored token prompts HERE and a cancel throws before the
  // POST. A flag set before calling loadModel would already be true.
  const { module, order } = chatApi({ proceed: false });
  await assert.rejects(
    module.loadModel(
      { model_path: "m" },
      { onRequestStart: () => order.push("announced") },
    ),
    (err: Error & { unslothUserCancelled?: boolean }) => {
      assert.equal(err.unslothUserCancelled, true);
      return true;
    },
  );
  assert.deepEqual(
    order,
    ["token"],
    "announced or sent a request it never made",
  );
});

test("an abort that lands during token preparation never announces a request", async () => {
  const controller = new AbortController();
  const { module, order } = chatApi({ proceed: true, token: null });
  controller.abort(new Error("Cancelled"));
  await assert.rejects(
    module.loadModel(
      { model_path: "m" },
      {
        signal: controller.signal,
        onRequestStart: () => order.push("announced"),
      },
    ),
  );
  assert.deepEqual(order, ["token"]);
});
