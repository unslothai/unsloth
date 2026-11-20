// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import * as generationLength from "../src/features/chat/api/generation-length.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// Read as source: importing would drag in the stores and the toast layer.
const source = readSrc("features/chat/api/chat-adapter.ts");

// Hoisted: biome's useTopLevelRegex flags a literal recompiled per call.
const TOAST_BRANCH =
  /err instanceof GenerationLengthError\) \{[\s\S]*?\n {10}\} else if/;
const FROM_THE_ERROR = /description:\s*\n?\s*msg \|\|/;
const THE_ERROR_CLASS = /class GenerationLengthError/;
const CAP_REMEDY = /Increase Max Tokens or disable thinking/;
const WINDOW_REMEDY = /cannot create room the window does not have/;
const NO_UNLIMITED_CLAIM = /already unlimited/;
const WINDOW_SETTING = /Context in Model settings/;
const HIDDEN_WINDOW_REMEDY =
  /Start a new chat, or shorten this one, then retry/;
const BOTH_REMEDIES = /Max Tokens or its context window before answering/;
const DURABLE_STREAM =
  /const durableStream = async function\* \(\) \{[\s\S]*?\n {12}\};/;
const DURABLE_TRACKER =
  /createReasoningOnlyLengthCheck\(\{\s*cap: requestPayload\.max_tokens \?\? null,\s*contextLength: servedContextLength,/;
const DURABLE_OBSERVE = /lengthCheck\.observe\(chunk\);\s*yield chunk;/;
const DURABLE_THROW =
  /generationStatus === "completed"\) \{\s*lengthCheck\.throwIfReasoningOnlyLength\(\);/;

type LengthCheck = {
  observe: (chunk: unknown) => void;
  throwIfReasoningOnlyLength: () => void;
};

// The real tracker and error; only chat-api's unrelated imports are stubbed.
const { createReasoningOnlyLengthCheck, GenerationLengthError } = loadWithStubs<{
  createReasoningOnlyLengthCheck: (window: {
    cap: number | null;
    contextLength: number | null;
    kvCacheHint?: boolean;
  }) => LengthCheck;
  GenerationLengthError: new (cause: string) => Error & { stopCause: string };
}>(new URL("../src/features/chat/api/chat-api.ts", import.meta.url), {
  "./skill-load-event": { skillLoadCardEvent: () => ({}) },
  "./thread-read-cache.ts": {},
  "@/features/auth": { authFetch: async () => new Response(null) },
  "@/lib/format-fastapi-error": { formatApiErrorBody: () => null },
  "@/lib/sse-framing": {},
  "../types": {},
  "../types/api": {},
  "../utils/chat-history-revision": {},
  "../utils/openapi-support": {},
  "../utils/load-warning-toast": {},
  "../lib/load-request": {},
  "./generation-length.ts": generationLength,
  "./gguf-variants-request": {},
  "./padded-response": {},
  "@/features/hf-auth": {},
  "@/features/igpu-carveout": {},
  "@/features/hub/lib/abort-signals": {},
  "@/features/hub/lib/hub-token-header": {},
  "@/features/hub/lib/network": {},
  "@/features/native-intents/api": {},
  "@/features/settings/low-disk-check": {},
  "@/lib/model-lifecycle-events": {},
});

// The Strix Halo repro: Qwen3-0.6B, Context 1024, a 43-token prompt, Max Tokens 512.
function reasoningOnlyRun(
  finish: string,
  answer?: string,
  window: { cap: number | null; contextLength: number | null; kvCacheHint?: boolean } = {
    cap: 512,
    contextLength: 1024,
  },
): LengthCheck {
  const check = createReasoningOnlyLengthCheck(window);
  check.observe({ usage: { prompt_tokens: 43 } });
  check.observe({ choices: [{ delta: { reasoning_content: "Primes below" } }] });
  check.observe({ choices: [{ delta: { reasoning_content: " 5000..." } }] });
  if (answer) check.observe({ choices: [{ delta: { content: answer } }] });
  check.observe({ choices: [{ delta: { content: null }, finish_reason: finish }] });
  return check;
}

test("the toast repeats the advice the error chose, not the Max Tokens advice", () => {
  const branch = TOAST_BRANCH.exec(source);
  assert.ok(branch, "the GenerationLengthError toast branch moved");
  assert.match(branch[0], FROM_THE_ERROR);
});

test("the two remedies really are different text, so passing it through matters", () => {
  const chatApi = readSrc("features/chat/api/chat-api.ts");

  assert.match(chatApi, THE_ERROR_CLASS);
  assert.match(chatApi, CAP_REMEDY);
  assert.match(chatApi, WINDOW_REMEDY);
  assert.match(chatApi, WINDOW_SETTING);
  assert.doesNotMatch(chatApi, NO_UNLIMITED_CLAIM);
  assert.match(chatApi, HIDDEN_WINDOW_REMEDY);
  assert.match(chatApi, BOTH_REMEDIES);
});

test("thinking that runs out of Max Tokens throws the Max Tokens advice", () => {
  assert.throws(
    () => reasoningOnlyRun("length").throwIfReasoningOnlyLength(),
    (err: unknown) =>
      err instanceof GenerationLengthError &&
      err.stopCause === "max_tokens" &&
      CAP_REMEDY.test(err.message),
  );
});

test("an answer, or a normal stop, is not a length failure", () => {
  assert.doesNotThrow(() =>
    reasoningOnlyRun("length", "669").throwIfReasoningOnlyLength(),
  );
  assert.doesNotThrow(() => reasoningOnlyRun("stop").throwIfReasoningOnlyLength());
});

test("server-side chat runs apply the same reasoning-only length check", () => {
  const durable = DURABLE_STREAM.exec(source);
  assert.ok(durable, "the durableStream generator moved");
  assert.match(durable[0], DURABLE_TRACKER);
  assert.match(durable[0], DURABLE_OBSERVE);
  assert.match(durable[0], DURABLE_THROW);
});

// Max Tokens unset: the backend sends the context length, and llama.cpp counts the thinking.
function windowFilled(kvCacheHint: boolean): Error & { stopCause: string } {
  const check = reasoningOnlyRun("length", undefined, {
    cap: 1024,
    contextLength: 1024,
    kvCacheHint,
  });
  check.observe({ timings: { predicted_n: 981 } });
  try {
    check.throwIfReasoningOnlyLength();
  } catch (err) {
    return err as Error & { stopCause: string };
  }
  throw new Error("no length error");
}

test("a llama.cpp model that fills its window is also pointed at a q8_0 KV cache", () => {
  const gguf = windowFilled(true);
  assert.equal(gguf.stopCause, "context_length");
  assert.match(gguf.message, /increase the Context in Model settings, or disable thinking/);
  assert.match(gguf.message, /q8_0 KV Cache in Model settings fits more context/);
  // NPU and provider models have no KV cache setting.
  assert.doesNotMatch(windowFilled(false).message, /KV Cache/);
  // Max Tokens advice never mentions it.
  assert.throws(
    () =>
      reasoningOnlyRun("length", undefined, {
        cap: 512,
        contextLength: 1024,
        kvCacheHint: true,
      }).throwIfReasoningOnlyLength(),
    (err: unknown) => err instanceof Error && !/KV Cache/.test(err.message),
  );
  assert.match(
    source,
    /const kvCacheHint = !isExternalRequest && runtime\.loadedIsGguf === true;/,
  );
});
