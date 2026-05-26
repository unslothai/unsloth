// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { matchesFormat } from "../src/features/models/lib/format-filters.ts";

test("checkpoint format filter includes only runnable checkpoint weight formats", () => {
  assert.equal(matchesFormat("safetensors", "checkpoint"), true);
  assert.equal(matchesFormat("checkpoint", "checkpoint"), true);
  assert.equal(matchesFormat(false, "checkpoint"), true);
  assert.equal(matchesFormat("adapter", "checkpoint"), false);
  assert.equal(matchesFormat("unknown", "checkpoint"), false);
  assert.equal(matchesFormat("gguf", "checkpoint"), false);
});

test("gguf format filter only includes gguf rows", () => {
  assert.equal(matchesFormat("gguf", "gguf"), true);
  assert.equal(matchesFormat(true, "gguf"), true);
  assert.equal(matchesFormat("safetensors", "gguf"), false);
});
