// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RUNTIME = readFileSync(
  fileURLToPath(new URL("../src/features/chat/hooks/use-chat-model-runtime.ts", import.meta.url)),
  "utf8",
);
const section = (start: string, end: string) => {
  const from = RUNTIME.indexOf(start);
  assert.notEqual(from, -1, `expected to find ${start}`);
  const to = RUNTIME.indexOf(end, from + start.length);
  assert.notEqual(to, -1, `expected to find ${end} after ${start}`);
  return RUNTIME.slice(from, to);
};

test("switching to a model with no saved settings loads it with the default KV cache type", () => {
  const reset = section("if (resetsPerModelSettings) {\n              loadSpeculativeType =", "\n            }\n");
  assert.match(reset, /loadKvCacheDtype = pendingLoadConfig\?\.kvCacheDtype \?\? null;/);
});

test("the already-loaded check compares a switch against the default KV cache type", () => {
  const mirror = section("const comparedConfig =", ": currentRuntimePerModelConfig());");
  assert.doesNotMatch(mirror, /kvCacheDtype/);
});
