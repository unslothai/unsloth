// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// saveChatMessage turns a protected 409 into a typed error so the autosave can tell "stop"
// from "retry". A manual edit rethrows to the user, so the server's wording must survive.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";
import { skillLoadCardEvent } from "../src/features/chat/api/skill-load-event.ts";
import * as abortSignals from "../src/features/hub/lib/abort-signals.ts";

type Module = {
  saveChatMessage: (
    message: Record<string, unknown>,
    options?: { allowGenerationEdit?: boolean; coalesce?: boolean },
  ) => Promise<unknown>;
  ChatMessageProtectedError: new (
    threadId: string,
    messageId: string,
    detail?: string,
  ) => Error & { threadId: string; messageId: string };
};

/** Minimal Response double: records how many times the body was consumed. */
function jsonResponse(status: number, body: unknown, kind: string | null = "protected") {
  let reads = 0;
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: (name: string) => (name === "X-Unsloth-Conflict-Kind" ? kind : null) },
    get bodyReads() {
      return reads;
    },
    async json() {
      reads += 1;
      if (reads > 1) {
        throw new TypeError("Body has already been consumed.");
      }
      if (body === undefined) throw new SyntaxError("Unexpected end of JSON input");
      return body;
    },
  };
}

function harness(
  response: ReturnType<typeof jsonResponse>,
  fetch?: (init?: RequestInit) => Promise<typeof response>,
  timeout = abortSignals.disposableTimeoutSignal,
) {
  const requests: { url: string; init?: RequestInit }[] = [];
  const module = loadWithStubs<Module>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "./skill-load-event": { skillLoadCardEvent },
      "./thread-read-cache.ts": loadWithStubs(
        new URL("../src/features/chat/api/thread-read-cache.ts", import.meta.url),
        {},
      ),
      "@/features/auth": {
        authFetch: async (url: string, init?: RequestInit) => {
          requests.push({ url, init });
          return fetch ? fetch(init) : response;
        },
      },
      "@/lib/format-fastapi-error": {
        formatApiErrorBody: (body: unknown) =>
          (body as { detail?: string } | null)?.detail ?? null,
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
      "./padded-response": { assertCompletedPaddedBody: () => {} },
      "@/features/hf-auth": { prepareHfTokenForUse: async () => undefined },
      // loadModel calls the notice on every load, and this harness names chat-api's
      // imports exactly, so the stub must exist even though no toast is raised here.
      "@/features/igpu-carveout": {
        dismissCarveoutAdviceForModel: () => {},
        showCarveoutAdvice: () => {},
      },
      "@/features/hub/lib/abort-signals": { ...abortSignals, disposableTimeoutSignal: timeout },
      "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
      "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
      "@/features/native-intents/api": { consumeNativePathToken: () => undefined },
      // loadModel reads the disk on the way in and out: a model the backend has to download
      // writes to the cache inside that request, passing no download-manager funnel.
      "@/features/settings/low-disk-check": { checkDiskSpace: () => Promise.resolve() },
      "@/lib/model-lifecycle-events": {},
    },
  );
  return { module, requests };
}

const message = { id: "m1", threadId: "t1", role: "assistant", content: [], createdAt: 1 };

test("a 409 becomes the typed error, carrying the server's wording", async () => {
  const response = jsonResponse(409, {
    detail: "server-managed generation messages cannot be edited",
  });
  const { module } = harness(response);

  const error = await module
    .saveChatMessage(message)
    .then(() => null)
    .catch((e) => e);

  assert.ok(
    error instanceof module.ChatMessageProtectedError,
    "callers branch on the type, so instanceof must hold",
  );
  assert.equal(
    error.message,
    "server-managed generation messages cannot be edited",
    "a manual edit surfaces this to the user; it must not be replaced",
  );
  assert.equal(error.threadId, "t1");
  assert.equal(error.messageId, "m1");
});

test("the body is read exactly once", async () => {
  const response = jsonResponse(409, { detail: "nope" });
  const { module } = harness(response);
  await module.saveChatMessage(message).catch(() => {});
  assert.equal(response.bodyReads, 1);
});

test("a 409 with no usable body still produces a sensible message", async () => {
  const { module } = harness(jsonResponse(409, null));
  const error = await module.saveChatMessage(message).catch((e) => e);
  assert.ok(error instanceof module.ChatMessageProtectedError);
  assert.match(error.message, /server-managed/);
});

test("a 409 whose body is not JSON at all does not throw a parse error", async () => {
  const { module } = harness(jsonResponse(409, undefined));
  const error = await module.saveChatMessage(message).catch((e) => e);
  assert.ok(
    error instanceof module.ChatMessageProtectedError,
    "the conflict must survive an unparseable body",
  );
});

test("a thread-id collision is not treated as protection", async () => {
  const { module } = harness(
    jsonResponse(409, { detail: "Message id already belongs to another thread: m1" }, "thread-collision"),
  );
  const error = await module.saveChatMessage(message).catch((e) => e);

  assert.ok(error instanceof Error);
  assert.ok(
    !(error instanceof module.ChatMessageProtectedError),
    "the caller must see this one and handle it",
  );
});

test("a backend too old to send the header falls back to the plain error", async () => {
  const { module } = harness(jsonResponse(409, { detail: "conflict" }, null));
  const error = await module.saveChatMessage(message).catch((e) => e);

  assert.ok(error instanceof Error);
  assert.ok(!(error instanceof module.ChatMessageProtectedError));
});

test("other failures are untouched", async () => {
  const { module } = harness(jsonResponse(500, { detail: "boom" }));
  const error = await module.saveChatMessage(message).catch((e) => e);
  assert.ok(error instanceof Error);
  assert.ok(
    !(error instanceof module.ChatMessageProtectedError),
    "only 409 means the server owns the message",
  );
  assert.match(error.message, /boom/);
});

test("a success is returned unchanged", async () => {
  const saved = { ...message, content: [{ type: "text", text: "hi" }] };
  const { module, requests } = harness(jsonResponse(200, saved));
  assert.deepEqual(await module.saveChatMessage(message), saved);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].init?.method, "PUT");
});

test("the manual-edit query parameter is still sent", async () => {
  const { module, requests } = harness(jsonResponse(200, message));
  await module.saveChatMessage(message, { allowGenerationEdit: true });
  assert.match(requests[0].url, /allowGenerationEdit=true/);
});

test("a stalled manual edit aborts and disposes its timeout", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let disposed = false;
  const timeout: typeof abortSignals.disposableTimeoutSignal = (ms) => {
    assert.equal(ms, 30_000);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new DOMException("timeout", "TimeoutError")), ms);
    return { signal: controller.signal, dispose: () => { disposed = true; clearTimeout(timer); } };
  };
  const { module, requests } = harness(jsonResponse(200, message), (init) => new Promise((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
  }), timeout);
  const result = assert.rejects(module.saveChatMessage(message, { allowGenerationEdit: true }), { name: "TimeoutError" });
  assert.equal(requests[0].init?.signal?.aborted, false);
  t.mock.timers.tick(30_000);
  await result;
  assert.equal(disposed, true);
});

test("a manual edit keeps its deadline while reading the response body", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const response = jsonResponse(200, message);
  let signal: AbortSignal | undefined;
  let disposed = false;
  response.json = () => new Promise((_resolve, reject) => {
    signal?.addEventListener("abort", () => reject(signal?.reason), { once: true });
  });
  const { module } = harness(response, async (init) => {
    signal = init?.signal ?? undefined;
    return response;
  }, (ms) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new DOMException("timeout", "TimeoutError")), ms);
    return { signal: controller.signal, dispose: () => { disposed = true; clearTimeout(timer); } };
  });
  const save = module.saveChatMessage(message, { allowGenerationEdit: true });
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(disposed, false);
  t.mock.timers.tick(30_000);
  await assert.rejects(save);
  assert.equal(disposed, true);
});
