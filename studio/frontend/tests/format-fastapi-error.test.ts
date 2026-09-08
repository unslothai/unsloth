// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  formatApiErrorBody,
  readFastApiError,
} from "../src/lib/format-fastapi-error.ts";

test("formats an OpenAI-compatible error envelope", () => {
  assert.equal(
    formatApiErrorBody({
      error: {
        message: "Audio file is too large (max ~25 MB).",
        type: "invalid_request_error",
      },
    }),
    "Audio file is too large (max ~25 MB).",
  );
});

test("formats an Anthropic-compatible error envelope", () => {
  assert.equal(
    formatApiErrorBody({
      type: "error",
      error: { type: "rate_limit_error", message: "Try again later." },
    }),
    "Try again later.",
  );
});

test("keeps FastAPI detail and top-level message support", () => {
  assert.equal(
    formatApiErrorBody({ detail: "Invalid request" }),
    "Invalid request",
  );
  assert.equal(
    formatApiErrorBody({ message: "Provider failed" }),
    "Provider failed",
  );
  assert.equal(
    formatApiErrorBody({
      detail: [{ loc: ["body", "messages"], msg: "Field required" }],
    }),
    "messages: Field required",
  );
});

test("unwraps an envelope nested in FastAPI's detail", () => {
  const flat = { error: { message: "Audio file is too large (max ~25 MB)." } };
  assert.equal(
    formatApiErrorBody({ detail: flat }),
    "Audio file is too large (max ~25 MB).",
  );
  assert.equal(formatApiErrorBody(flat), formatApiErrorBody({ detail: flat }));
  assert.equal(
    formatApiErrorBody({
      detail: {
        error: {
          message: "n > 1 is not supported for GGUF tool chat completions.",
          code: "unsupported_parameter",
          param: "n",
        },
      },
    }),
    "n > 1 is not supported for GGUF tool chat completions.",
  );
  assert.equal(
    formatApiErrorBody({ detail: { message: "Nested plain message" } }),
    "Nested plain message",
  );
});

test("prefers a string detail over anything nested under it", () => {
  assert.equal(
    formatApiErrorBody({ detail: "Audio is too large." }),
    "Audio is too large.",
  );
  assert.equal(formatApiErrorBody({ detail: {} }), null);
  assert.equal(formatApiErrorBody({ detail: { error: {} } }), null);
  assert.equal(formatApiErrorBody({ detail: [] }), null);
});

test("reads an envelope nested in detail off a response", async () => {
  const response = new Response(
    JSON.stringify({
      detail: { error: { message: "Audio file is too large (max ~25 MB)." } },
    }),
    { status: 413, headers: { "Content-Type": "application/json" } },
  );
  assert.equal(
    await readFastApiError(response),
    "Audio file is too large (max ~25 MB).",
  );
});

test("does not recurse without bound on a deeply nested detail", () => {
  let body: unknown = { error: { message: "too deep to reach" } };
  for (let i = 0; i < 50; i++) body = { detail: body };
  assert.equal(formatApiErrorBody(body), null);
});

test("reads an OpenAI-compatible error response", async () => {
  const response = new Response(
    JSON.stringify({ error: { message: "Context limit exceeded" } }),
    { status: 400, headers: { "Content-Type": "application/json" } },
  );
  assert.equal(await readFastApiError(response), "Context limit exceeded");
});

test("falls back for malformed or empty error bodies", async () => {
  for (const body of [
    null,
    {},
    { error: null },
    { error: {} },
    { error: { message: 3 } },
  ]) {
    assert.equal(formatApiErrorBody(body), null);
  }

  const response = new Response("not JSON", { status: 503 });
  assert.equal(await readFastApiError(response, "HTTP"), "HTTP (503)");
});
