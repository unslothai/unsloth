// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Padded load/unload commit a 200 early; a killed tunnel yields a 200 with an empty body,
 * which `catch(() => null)` would read as success.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { assertCompletedPaddedBody } from "../src/features/chat/api/padded-response.ts";

import { readSrc, readText } from "./helpers/kit.ts";

const chatApi = readSrc("features/chat/api/chat-api.ts");

test("a real payload passes through", () => {
  assertCompletedPaddedBody({ status: "loaded", model: "org/A" }, "Model load");
  assertCompletedPaddedBody({ status: "unloaded" }, "Model unload");
});

test("a body the proxy truncated is rejected, not accepted as success", () => {
  // An empty body, a pad-only body and a half payload all decode to null.
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
  const cli = readText("../../../unsloth_cli/_inference.py");
  assert.ok(cli.includes("def require_completed_padded_body("));
  assert.ok(cli.includes("if isinstance(body, dict) and body:"));
});
