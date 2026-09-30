// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { mlxSpecFallbackMessage } from "../src/features/chat/lib/mlx-spec-fallback.ts";

test("an MLX fallback notice says what runs in the passed-over drafter's place", () => {
  assert.match(mlxSpecFallbackMessage("drafter_not_found", "mtp") ?? "", /with MTP instead/);
  assert.match(mlxSpecFallbackMessage("drafter_no_memory", "ngram") ?? "", /n-gram copies only/);
  assert.match(mlxSpecFallbackMessage("auto_context_cost", null) ?? "", /without speculative/);
  // llama.cpp's codes are not MLX's, so an unknown one shows nothing rather than a wrong remedy.
  assert.equal(mlxSpecFallbackMessage("binary_no_mtp", null), null);
});

test("an MLX fallback notice claims no more than every path behind its code", () => {
  // No code guarantees its remedy works, and drafter_no_memory also covers an unpriced fit.
  for (const code of ["drafter_not_found", "drafter_incompatible", "drafter_no_memory", "runtime_error", "kv_quant", "auto_context_cost", "auto_span_drafter"]) {
    assert.doesNotMatch(mlxSpecFallbackMessage(code, null) ?? "missing", /to use it|missing/);
  }
  assert.doesNotMatch(mlxSpecFallbackMessage("drafter_no_memory", null) ?? "", /does not fit/);
});
