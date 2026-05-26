// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  ggufVariantFromStorageKey,
  ggufVariantsMatch,
  modelIdFromStorageKey,
  modelIdsMatch,
  modelStorageKey,
  normalizeGgufVariantIdentity,
} from "../src/lib/model-identity.ts";

test("model identity matches Hugging Face ids case-insensitively", () => {
  assert.equal(
    modelIdsMatch("Unsloth/Qwen3-8B-GGUF", "unsloth/qwen3-8b-gguf"),
    true,
  );
});

test("GGUF variant identity is trimmed and case folded", () => {
  assert.equal(normalizeGgufVariantIdentity(" Q4_K_M "), "q4_k_m");
  assert.equal(ggufVariantsMatch("Q4_K_M", "q4_k_m"), true);
});

test("model storage keys normalize GGUF variant casing", () => {
  const key = modelStorageKey("Org/Foo", "Q4_K_M");
  assert.equal(
    key,
    modelStorageKey("org/foo", "q4_k_m"),
  );
  assert.equal(key.startsWith("v2:"), true);
  assert.equal(modelIdFromStorageKey(key), "org/foo");
  assert.equal(ggufVariantFromStorageKey(key), "q4_k_m");
});

test("model storage keys isolate separator-bearing ids and variants", () => {
  const first = modelStorageKey("/models/a::b", "Q4_K_M");
  const second = modelStorageKey("/models/a", "b::Q4_K_M");

  assert.notEqual(first, second);
  assert.equal(modelIdFromStorageKey(first), "/models/a::b");
  assert.equal(ggufVariantFromStorageKey(first), "q4_k_m");
  assert.equal(modelIdFromStorageKey(second), "/models/a");
  assert.equal(ggufVariantFromStorageKey(second), "b::q4_k_m");
});
