// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  lengthStopCause,
  maxTokensIsTheLimit,
  windowEvidenceCount,
} from "../src/features/chat/api/generation-length.ts";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { externalStopWindow } = await import(
  "../src/features/chat/provider-capabilities.ts"
);

const CHAT_ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const CHAT_API = readSrc("features/chat/api/chat-api.ts");

// Hoisted: biome's useTopLevelRegex flags a literal recompiled per call.
const LOCAL_WINDOW_ARGUMENT =
  /isExternalRequest\s*\n\s*\? externalStopWindow\(\s*externalProvider\?\.providerType,\s*externalProvider\?\.baseUrl,\s*\)\s*\n\s*: \(runtime\.loadedCustomContextLength \?\?\s*\n\s*runtime\.loadedContextLength \?\?\s*\n\s*\(params\.maxSeqLength \|\| null\)\)/;
// A pending context edit is `customContextLength !== loadedCustomContextLength`.
const PENDING_FIELD = /: \(runtime\.customContextLength \?\?/;
const WINDOW_COUNT_FROM_TIMINGS = /windowEvidenceCount\(parsedTimings\)/;
const ADAPTER_COUNT_FROM_TIMINGS = /windowEvidenceCount\(chunkTimings\)/;
const PROVIDER_WINDOW_EVENT =
  /parsedToolEvent\?\.type === "context_window_exceeded"/;
const PROVIDER_WINDOW_WINS = /providerReportedWindow\s*\?\s*"context_window"/;

test("a cap the prompt left no room for is not the limit that was hit", () => {
  assert.equal(
    maxTokensIsTheLimit({ cap: 2048, contextLength: 4096, promptTokens: 3000 }),
    false,
  );
});

test("a cap the prompt left room for is the limit that was hit", () => {
  assert.equal(
    maxTokensIsTheLimit({ cap: 512, contextLength: 4096, promptTokens: 3000 }),
    true,
  );
});

test("hitting the cap and the context wall together is context-bound", () => {
  // At equality both limits are hit on the same token, so Context Length is the remedy.
  assert.equal(
    maxTokensIsTheLimit({ cap: 1096, contextLength: 4096, promptTokens: 3000 }),
    false,
  );
  assert.equal(
    maxTokensIsTheLimit({ cap: 1095, contextLength: 4096, promptTokens: 3000 }),
    true,
  );
});

test("Max Tokens on Max is never the limit", () => {
  // The backend substitutes the context length, so a cap equal to it is the same as unset.
  assert.equal(
    maxTokensIsTheLimit({ cap: 4096, contextLength: 4096, promptTokens: 10 }),
    false,
  );
  assert.equal(
    maxTokensIsTheLimit({ cap: null, contextLength: 4096, promptTokens: 10 }),
    false,
  );
});

test("without a prompt count the cap alone decides", () => {
  assert.equal(
    maxTokensIsTheLimit({ cap: 2048, contextLength: 4096, promptTokens: null }),
    true,
  );
});

test("an unknown context length cannot make a cap the limit", () => {
  assert.equal(
    maxTokensIsTheLimit({ cap: 2048, contextLength: null, promptTokens: null }),
    true,
  );
  assert.equal(
    maxTokensIsTheLimit({ cap: null, contextLength: null, promptTokens: null }),
    false,
  );
});

test("a local model with no GGUF window still reports one", () => {
  // On the legacy path params.maxSeqLength is the effective window; null would make it infinite.
  assert.equal(
    maxTokensIsTheLimit({ cap: 2048, contextLength: null, promptTokens: 3000 }),
    true,
  );
  assert.equal(
    maxTokensIsTheLimit({ cap: 2048, contextLength: 4096, promptTokens: 3000 }),
    false,
  );

  assert.match(CHAT_ADAPTER, LOCAL_WINDOW_ARGUMENT);
});

test("a pending Context Length edit does not decide what stopped the generation", () => {

  assert.match(CHAT_ADAPTER, LOCAL_WINDOW_ARGUMENT);
  assert.doesNotMatch(CHAT_ADAPTER, PENDING_FIELD);
});

test("a stop short of the cap against a window Studio cannot see filled that window", () => {
  assert.equal(
    lengthStopCause({
      cap: 2048,
      contextLength: null,
      promptTokens: 3857,
      completionTokens: 239,
    }),
    "context_window",
  );
  assert.equal(
    lengthStopCause({
      cap: 2048,
      contextLength: null,
      promptTokens: null,
      completionTokens: 2048,
    }),
    "max_tokens",
  );
});

test("with no window and no output count, neither wall is claimed", () => {
  assert.equal(
    lengthStopCause({
      cap: 2048,
      contextLength: null,
      promptTokens: null,
      completionTokens: null,
    }),
    "unknown",
  );
});

test("a known window still decides from the prompt, as before", () => {
  assert.equal(
    lengthStopCause({
      cap: 2048,
      contextLength: 4096,
      promptTokens: 3000,
      completionTokens: 1096,
    }),
    "context_length",
  );
  assert.equal(
    lengthStopCause({
      cap: 512,
      contextLength: 4096,
      promptTokens: 3000,
      completionTokens: 512,
    }),
    "max_tokens",
  );
});

function causeOf(
  chunk: {
    usage?: { prompt_tokens?: number; completion_tokens?: number };
    timings?: { predicted_n?: number };
  },
  cap: number,
) {
  return lengthStopCause({
    cap,
    contextLength: null,
    promptTokens: chunk.usage?.prompt_tokens ?? null,
    completionTokens: windowEvidenceCount(chunk.timings),
  });
}

test("llama.cpp's own count tells a full window from Max Tokens", () => {
  assert.equal(
    causeOf({ timings: { predicted_n: 254 } }, 2048),
    "context_window",
  );
  assert.equal(causeOf({ timings: { predicted_n: 2048 } }, 2048), "max_tokens");
  assert.equal(windowEvidenceCount({ predicted_n: "254" }), null);
});

test("a count that leaves out reasoning does not claim a full window", () => {
  assert.equal(
    causeOf(
      {
        usage: { prompt_tokens: 32, completion_tokens: 9 },
      },
      64,
    ),
    "unknown",
  );
});

test("an output budget the server lowered does not claim a full window", () => {
  assert.equal(
    causeOf({ usage: { prompt_tokens: 900, completion_tokens: 4096 } }, 8192),
    "unknown",
  );
});

test("the stream reads the window from llama.cpp timings and the provider's own report", () => {
  assert.match(CHAT_API, WINDOW_COUNT_FROM_TIMINGS);
  assert.match(CHAT_ADAPTER, ADAPTER_COUNT_FROM_TIMINGS);
  assert.match(CHAT_API, PROVIDER_WINDOW_EVENT);
  assert.match(CHAT_API, PROVIDER_WINDOW_WINS);
});

test("a Gemini length stop is always the output limit", () => {
  const window = externalStopWindow("gemini", null);
  for (const [completionTokens, cap] of [
    [60, 64],
    [99, 100],
    [8177, 8192],
  ]) {
    assert.equal(
      lengthStopCause({
        cap,
        contextLength: window,
        promptTokens: 1200,
        completionTokens,
      }),
      "max_tokens",
    );
  }
  assert.equal(
    externalStopWindow(
      "gemini",
      "https://generativelanguage.googleapis.com/v1beta",
    ),
    Number.POSITIVE_INFINITY,
  );
  assert.equal(externalStopWindow("gemini", "http://localhost:4000/v1"), null);
  assert.equal(externalStopWindow("custom", null), null);
  assert.equal(externalStopWindow(undefined, null), null);
});
