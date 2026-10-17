// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const DETAIL = {
  code: "out_of_memory",
  message: "Not enough GPU memory to load Qwen3-TTS.",
  smaller_variants: [{ variant: "Q4_K_M", label: "Q4_K_M" }],
};

let reply: Response = new Response("{}");
const api = loadWithStubs<typeof import("../src/features/audio/api.ts")>(
  new URL("../src/features/audio/api.ts", import.meta.url),
  {
    "@/features/auth": {
      authFetch: async () => reply,
      getAuthToken: () => "t",
    },
    "@/lib/account-transition": { accountTransitionPending: () => false },
    "@/lib/api-base": { apiUrl: (path: string) => path },
    "@/lib/format-fastapi-error": {
      formatApiErrorBody: () => null,
      readFastApiError: async (response: Response) =>
        ((await response.json()) as { detail: { message: string } }).detail
          .message,
    },
    "./audio-run-request": {
      buildAudioRunBody: () => ({}),
      buildVoiceCreateBody: () => ({}),
      transcribeUrl: () => "/t",
    },
  },
);

test("AudioApiError keeps the detail and its code; a string detail has no code", () => {
  const structured = new api.AudioApiError("m", 503, DETAIL);
  assert.deepEqual(
    [structured.code, structured.detail],
    ["out_of_memory", DETAIL],
  );
  assert.equal(new api.AudioApiError("m", 501, "plain").code, undefined);
});

test("generateAudio and runAudio throw AudioApiError with the server's message, status and code", async () => {
  for (const call of [
    () => api.generateAudio("hi"),
    () => api.runAudio({ workflow: "speak", text: "hi" }),
  ]) {
    reply = new Response(JSON.stringify({ detail: DETAIL }), { status: 503 });
    await assert.rejects(call(), (error: unknown) => {
      assert.ok(error instanceof api.AudioApiError);
      assert.deepEqual(
        [error.message, error.status, error.code],
        [DETAIL.message, 503, "out_of_memory"],
      );
      return true;
    });
  }
});

test("chat loads throw LoadModelError with the detail, for both an error status and _deferred_error", () => {
  const chat = readSrc("features/chat/api/chat-api.ts");
  assert.match(chat, /export class LoadModelError extends Error \{/);
  assert.match(
    chat,
    /throw new LoadModelError\(parseErrorText\(response\.status, body\), response\.status, detail\)/,
  );
  assert.match(
    chat,
    /throw new LoadModelError\(deferred\.message, deferred\.status, deferred\.detail\)/,
  );
  assert.match(chat, /detail: deferred\.detail,/);
  assert.doesNotMatch(chat, /throw new Error\(deferred\.message\)/);
});
