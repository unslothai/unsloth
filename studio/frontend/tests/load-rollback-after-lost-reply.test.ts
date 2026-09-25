// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A model switch whose /api/inference/load got no answer must not load the previous model back.
//
// Switching (or reloading the same model with new settings) unloads the resident model first, and a
// failed load then restores it. That is right when the backend REPORTED the failure. It is wrong when
// the connection closed before any answer, because the backend keeps loading after its client goes
// away: the rollback, sent with force_cancel_active, queued behind the load and replaced it.
//
// Reloading the page mid-load is the everyday way to get there. Chromium and Firefox both reject the
// old document's pending fetch and still let its catch send one more request. Seen in the Chat UI
// extra lane (playwright_model_config.py, "context length 4096 persists"): the 4096 load finished,
// the dying page's rollback then loaded the model back at its previous 2048, and the reloaded page
// correctly declined to show the remembered 4096 for a server running 2048, so it read "Auto".
//
// The rollback lives inside performLoad with no seam to call, so the guard is checked at the source,
// and everything that feeds it (authFetch, loadModel, the padded-body check) runs for real against a
// loopback server that drops the connection the way a page reload does.

import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const paddedResponse = await import("../src/features/chat/api/padded-response.ts");
const { shouldRestorePreviousModel, loadOutcomeUnknown } = await import(
  "../src/features/chat/lib/restore-previous-model.ts"
);

const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");

type Reply = "drop" | "padded-drop" | "rejected" | "deferred" | "loaded";

type AuthApi = {
  authFetch: (input: string, init?: RequestInit) => Promise<Response>;
};
type ChatApi = {
  loadModel: (payload: Record<string, unknown>) => Promise<Record<string, unknown>>;
};

async function withLoadServer(
  reply: Reply,
  run: (port: number) => Promise<void>,
): Promise<void> {
  const server: Server = createServer((req: IncomingMessage, res: ServerResponse) => {
    req.resume();
    req.on("end", () => {
      if (reply === "drop") {
        // The request arrived and the reply never will: what the old page sees on reload.
        req.socket.destroy();
        return;
      }
      if (reply === "padded-drop") {
        // A load past _TUNNEL_KEEPALIVE_AFTER_S has committed its 200 and is padding when the page goes.
        res.writeHead(200, { "Content-Type": "application/json" });
        res.write("   ");
        setTimeout(() => req.socket.destroy(), 20);
        return;
      }
      const body =
        reply === "rejected"
          ? { detail: "Not enough memory to load this model." }
          : reply === "deferred"
            ? { _deferred_error: { status_code: 500, detail: "llama-server exited" } }
            : { status: "loaded", model: "unsloth/gemma-3-270m-it-GGUF" };
      res.writeHead(reply === "rejected" ? 500 : 200, {
        "Content-Type": "application/json",
      });
      res.end(JSON.stringify(body));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("no port");
  try {
    await run(address.port);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

function realAuthFetch(port: number): AuthApi {
  return loadWithStubs<AuthApi>(new URL("../src/features/auth/api.ts", import.meta.url), {
    "@/lib/api-base": {
      apiUrl: (path: string) => `http://127.0.0.1:${port}${path}`,
      getApiPort: () => port,
      isTauri: false,
    },
    "@/lib/account-transition": { accountTransitionPending: () => false },
    "./session": {
      clearAuthTokens: () => {},
      getAuthToken: () => "access-token",
      getRefreshToken: () => null,
      mustChangePassword: () => false,
      setMustChangePassword: () => {},
      storeAuthTokens: () => {},
    },
    "@tauri-apps/api/core": {
      invoke: async () => {
        throw new Error("not Tauri");
      },
    },
  });
}

function realLoadModel(auth: AuthApi): ChatApi {
  return loadWithStubs<ChatApi>(new URL("../src/features/chat/api/chat-api.ts", import.meta.url), {
    "@/features/auth": { authFetch: auth.authFetch },
    "./padded-response": paddedResponse,
    "@/lib/format-fastapi-error": {
      formatApiErrorBody: (body: { detail?: unknown } | null) =>
        typeof body?.detail === "string" ? body.detail : null,
    },
    "../types": {},
    "../types/api": {},
    "../utils/chat-history-revision": {
      notifyChatHistoryUpdated: () => {},
      isCoalescedHistoryEvent: () => false,
    },
    "../utils/load-warning-toast": { showLoadWarning: () => {} },
    "./generation-length.ts": {},
    "./gguf-variants-request": {},
    "@/features/hf-auth": { prepareHfTokenForUse: async () => ({ proceed: true }) },
    "@/features/igpu-carveout": {
      dismissCarveoutAdviceForModel: () => {},
      showCarveoutAdvice: () => {},
    },
    "@/features/hub/lib/abort-signals": {},
    "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
    "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
    "@/features/native-intents/api": { consumeNativePathToken: () => undefined },
    "@/features/settings/low-disk-check": { checkDiskSpace: () => Promise.resolve() },
    "@/lib/model-lifecycle-events": {
      withModelLoadNotice: async (_r: string, _p: string | null, run: () => Promise<unknown>) =>
        run(),
    },
  });
}

/** What performLoad's catch receives when this load's reply is `reply`. */
async function loadError(reply: Reply): Promise<unknown> {
  let caught: unknown = null;
  await withLoadServer(reply, async (port) => {
    const { loadModel } = realLoadModel(realAuthFetch(port));
    try {
      await loadModel({
        model_path: "unsloth/gemma-3-270m-it-GGUF",
        gguf_variant: "UD-Q4_K_XL",
        max_seq_length: 4096,
      });
    } catch (error) {
      caught = error;
    }
  });
  return caught;
}

test("a load whose connection closed before any reply does not restore the previous model", async () => {
  const error = await loadError("drop");
  assert.ok(error instanceof Error, "the dropped load must reject");
  assert.equal(loadOutcomeUnknown(error), true);
  assert.equal(shouldRestorePreviousModel(error), false);
});

test("a padded load cut off mid-reply does not restore the previous model either", async () => {
  const error = await loadError("padded-drop");
  assert.ok(error instanceof Error, "the truncated load must reject");
  assert.match((error as Error).message, /Model load did not report completion/);
  assert.equal(loadOutcomeUnknown(error), true);
  assert.equal(shouldRestorePreviousModel(error), false);
});

test("a failure the backend reported still restores the previous model", async () => {
  for (const reply of ["rejected", "deferred"] as const) {
    const error = await loadError(reply);
    assert.ok(error instanceof Error, `${reply} must reject`);
    assert.equal(loadOutcomeUnknown(error), false, reply);
    assert.equal(shouldRestorePreviousModel(error), true, reply);
  }
});

test("a completed load is not an error at all", async () => {
  assert.equal(await loadError("loaded"), null);
});

test("non-errors and untagged errors keep the rollback", () => {
  for (const value of [null, undefined, "boom", 0, {}, new Error("x"), new TypeError("x")]) {
    assert.equal(shouldRestorePreviousModel(value), true, String(value));
  }
  assert.equal(
    shouldRestorePreviousModel(Object.assign(new Error("x"), { unslothTransportFailure: "yes" })),
    true,
  );
});

test("performLoad only rolls back when the failed load was answered", () => {
  // One rollback site, and its guard consults the helper before any request is sent.
  const sites = RUNTIME.match(/const rollbackResponse = await loadModel\(/g) ?? [];
  assert.equal(sites.length, 1);
  const guarded =
    /if \(\s*previousWasUnloaded &&\s*previousCheckpoint &&\s*shouldRestorePreviousModel\(error\)\s*\) \{[\s\S]*?const rollbackResponse = await loadModel\(/;
  assert.match(RUNTIME, guarded);
  // The failed load still surfaces: the guard skips the rollback, not the rethrow.
  const catchBlock = RUNTIME.slice(
    RUNTIME.indexOf("notifyLocalPromptQueueLoadFailed(lifecycleLease);"),
    RUNTIME.indexOf("const isCachedLoad = downloadComplete;"),
  );
  assert.match(catchBlock, /\}\s*throw error;\s*\}\s*\}\s*$/);
});
