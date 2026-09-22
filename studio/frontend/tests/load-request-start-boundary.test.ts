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
        authFetch: async (
          _input: unknown,
          _init: unknown,
          authOptions?: { onRequestStart?: () => void },
        ) => {
          // Where authFetch itself announces the send: past its own local refusals, which
          // is the boundary this contract is about.
          authOptions?.onRequestStart?.();
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


test("a local refusal inside authFetch never announces the send", async () => {
  // The refusal that has no request behind it at all: a peer tab is mid account switch, so
  // authFetch throws before any bytes leave. A caller that announced the send before calling
  // it then offered the server log for a load the backend never saw.
  const order: string[] = [];
  const auth = loadWithStubs<{
    authFetch: (
      input: string,
      init?: RequestInit,
      options?: { onRequestStart?: () => void },
    ) => Promise<unknown>;
  }>(new URL("../src/features/auth/api.ts", import.meta.url), {
    "@/lib/account-transition": { accountTransitionPending: () => true },
    "@/lib/api-base": { apiUrl: (path: string) => path, isTauri: false },
    "./session": {
      clearAuthTokens: () => {},
      getAuthToken: () => "access-token",
      getRefreshToken: () => null,
      mustChangePassword: () => false,
      setMustChangePassword: () => {},
      storeAuthTokens: () => {},
    },
  });

  await assert.rejects(
    auth.authFetch("/api/inference/load", { method: "POST" }, {
      onRequestStart: () => order.push("announced"),
    }),
    /switching accounts/,
  );
  assert.deepEqual(order, [], "announced a send that authFetch refused locally");
});


test("authFetch announces the send once, immediately before the transport", async () => {
  const order: string[] = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    order.push("fetch");
    return new Response(null, { status: 200 });
  };
  try {
    const auth = loadWithStubs<{
      authFetch: (
        input: string,
        init?: RequestInit,
        options?: { onRequestStart?: () => void },
      ) => Promise<unknown>;
    }>(new URL("../src/features/auth/api.ts", import.meta.url), {
      "@/lib/account-transition": { accountTransitionPending: () => false },
      "@/lib/api-base": { apiUrl: (path: string) => path, isTauri: false },
      "./session": {
        clearAuthTokens: () => {},
        getAuthToken: () => "access-token",
        getRefreshToken: () => null,
        mustChangePassword: () => false,
        setMustChangePassword: () => {},
        storeAuthTokens: () => {},
      },
    });

    await auth.authFetch("/api/inference/load", { method: "POST" }, {
      onRequestStart: () => order.push("announced"),
    });
    assert.deepEqual(
      order,
      ["announced", "fetch"],
      "authFetch does not announce the send it is about to make",
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
