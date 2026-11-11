// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("loadModel sends the caller's headers, so sweep loads carry the bench lease", () => {
  const src = readSrc("features/chat/api/chat-api.ts");
  const start = src.indexOf("export async function loadModel(");
  assert.ok(start >= 0, "loadModel moved");
  const body = src.slice(start, src.indexOf("\nexport ", start + 1));
  const call = body.slice(body.indexOf('authFetch("/api/inference/load"'));
  const headers = call.match(/headers:\s*\{([^}]*)\}/);
  assert.ok(headers, "load request headers not found");
  assert.match(headers[1], /\.\.\.options\?\.headers/);
  assert.match(headers[1], /"Content-Type": "application\/json"/);
  // The runner relies on this helper for every variant load and restore.
  assert.match(
    readSrc("features/benchmarks/api/bench-runner.ts"),
    /headers: leaseHeaders\(/,
  );
});
