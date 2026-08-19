// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A truncated padded reply is not a successful load.
 *
 * `/api/inference/load` and `/unload` pad their body so a proxy cannot time the request
 * out, which commits the 200 before the work finishes. The tunnel probe measured the
 * failure mode (studio/backend/tests/test_tunnel_safe_long_post.py): one byte at t=90s
 * then silence is killed ~125s later and the client sees a 200 with an EMPTY body.
 * `response.json()` throws on that and `catch(() => null)` makes it a 200 with a null
 * body, read as success unless the padded callers say otherwise.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { assertCompletedPaddedBody } from "../src/features/chat/api/padded-response.ts";

const chatApi = readFileSync(
  new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
  "utf8",
);

test("a real payload passes through", () => {
  assertCompletedPaddedBody({ status: "loaded", model: "org/A" }, "Model load");
  assertCompletedPaddedBody({ status: "unloaded" }, "Model unload");
});

test("a body the proxy truncated is rejected, not accepted as success", () => {
  // An empty body, a pad-only body ("   ") and a half payload all decode to null.
  for (const body of [null, undefined, {}, [], "", "loaded", 0]) {
    assert.throws(
      () => assertCompletedPaddedBody(body, "Model load"),
      /Model load did not report completion/,
      JSON.stringify(body) ?? "undefined",
    );
  }
});

test("the message names the operation and points at the model's status", () => {
  assert.throws(
    () => assertCompletedPaddedBody(null, "Model unload"),
    (err: unknown) => {
      const message = (err as Error).message;
      assert.match(message, /^Model unload did not report completion/);
      assert.match(message, /connection closed/);
      assert.match(message, /Check the model's status/);
      return true;
    },
  );
});

test("only the two padded routes require a payload", () => {
  // Scoped: parseJsonOrThrow serves ~30 endpoints, some legitimately with no body.
  const labelled = [
    ...chatApi.matchAll(/parseJsonOrThrow<[^>]*>\(\s*response,\s*"([^"]+)"/g),
  ].map((match) => match[1]);
  assert.deepEqual(labelled, ["Model load", "Model unload"]);
  assert.ok(chatApi.includes("assertCompletedPaddedBody(body, paddedLabel)"));
});

test("the Python client agrees", () => {
  // If only one client rejects a truncated reply, one reads success where the other fails.
  const cli = readFileSync(
    new URL("../../../unsloth_cli/_inference.py", import.meta.url),
    "utf8",
  );
  assert.ok(cli.includes("def require_completed_padded_body("));
  // unsloth_cli/tests/test_inference_chat.py asserts it actually raises.
  assert.ok(cli.includes("if isinstance(body, dict) and body:"));
});
