// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// saveChatMessage turns a protected 409 into a typed error so the autosave can tell "stop"
// from "retry". A manual edit rethrows to the user, so the server's wording must survive.

import assert from "node:assert/strict";
import test from "node:test";

import * as admissionStatus from "../src/features/chat/utils/admission-status.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

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

function harness(response: ReturnType<typeof jsonResponse>) {
  const requests: { url: string; init?: RequestInit }[] = [];
  const module = loadWithStubs<Module>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (url: string, init?: RequestInit) => {
          requests.push({ url, init });
          return response;
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
      // The real module, not a double: it is pure, imports nothing, and is only reached
      // from the SSE reader these tests never enter. Passing it through means this entry
      // cannot drift away from the implementation the way a hand-written stub would.
      "../utils/admission-status": admissionStatus,
      "./generation-length.ts": {},
      "./gguf-variants-request": {},
      "./padded-response": { assertCompletedPaddedBody: () => {} },
      "@/features/hf-auth": { prepareHfTokenForUse: async () => undefined },
      "@/features/hub/lib/abort-signals": {},
      "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
      "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
      "@/features/native-intents/api": { consumeNativePathToken: () => undefined },
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
