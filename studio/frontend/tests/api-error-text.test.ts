// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

let reply: Response = new Response("{}");
const authFetch = async () => reply;

function respond(status: number, body: unknown): void {
  reply = new Response(JSON.stringify(body), { status });
}

const prompts = loadWithStubs<
  typeof import("../src/features/chat/api/prompts-api.ts")
>(new URL("../src/features/chat/api/prompts-api.ts", import.meta.url), {
  "@/features/auth": { authFetch },
  "@/lib/format-fastapi-error": formatFastApiError,
});

const dictation = loadWithStubs<
  typeof import("../src/features/chat/adapters/studio-model-dictation-adapter.ts")
>(
  new URL(
    "../src/features/chat/adapters/studio-model-dictation-adapter.ts",
    import.meta.url,
  ),
  {
    "@/features/auth": { authFetch },
    "@/lib/format-fastapi-error": formatFastApiError,
    "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
    "@/lib/model-lifecycle-events": {},
    "../../audio/audio-cpp-catalog": {},
    "@/features/hub/inventory/api": {},
    "@/features/hub/stores/hf-token-store": {},
    "@/features/settings/stores/settings-dialog-store": {},
    "@/features/settings/stores/stt-download-prompt-store": {},
    "@/features/settings/stores/voice-settings-store": {},
    sonner: {},
    "../../hub/lib/abort-signals": {},
    "../api/providers-api": {},
    "../external-providers": {},
    "../stores/external-providers-store": {},
    "./dictation-level": {},
    "./pcm-recorder": {},
    "./stt-errors": {},
    "./dictation-outcome": {},
    "./studio-web-speech-dictation-adapter": {},
  },
);

const entry = {
  id: "p1",
  name: "Long prompt",
  text: "x",
  createdAt: 1,
  updatedAt: 1,
};

test("saving an over-limit prompt shows the 422 validation message", async () => {
  // FastAPI's body for PromptEntry.text = Field(max_length = 100_000).
  respond(422, {
    detail: [
      {
        type: "string_too_long",
        loc: ["body", "text"],
        msg: "String should have at most 100000 characters",
        ctx: { max_length: 100000 },
      },
    ],
  });
  await assert.rejects(prompts.savePromptEntry(entry), {
    message: "text: String should have at most 100000 characters",
  });
});

test("a prompt route's string detail and empty body keep their text", async () => {
  respond(400, { detail: "ID mismatch" });
  await assert.rejects(prompts.savePromptEntry(entry), {
    message: "ID mismatch",
  });
  reply = new Response("", { status: 500 });
  await assert.rejects(prompts.listPromptEntries(), {
    message: "Request failed (500)",
  });
});

test("a busy STT download reports the status, not [object Object]", async () => {
  respond(409, { detail: { error: "download_busy", retry_after: 1 } });
  await assert.rejects(
    dictation.startSttDownload("openai/whisper-small", undefined, "transformers"),
    { message: "HTTP 409" },
  );
});

test("STT download and cancel errors still show the server's string detail", async () => {
  respond(403, { detail: "Account is retired" });
  await assert.rejects(
    dictation.startSttDownload("openai/whisper-small", undefined, "transformers"),
    { message: "Account is retired" },
  );
  respond(409, { detail: "Another download is still running" });
  await assert.rejects(
    dictation.cancelSttDownload("openai/whisper-small", "transformers"),
    { message: "Another download is still running" },
  );
});
