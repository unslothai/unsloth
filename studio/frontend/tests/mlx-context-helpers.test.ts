// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Behaviour of the pure helpers the MLX context work introduced.
//
// The rest of the coverage for these reads their source with a regex, which passes
// unchanged if the body is deleted. These import them and call them, so a helper that
// stops answering fails here.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const {
  DEFAULT_MAX_SEQ_LENGTH,
  isServedByLlamaCpp,
  isServedByMlx,
  loadedContextFields,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const {
  capturedContextLength,
  loadedContextForParams,
  localMaxTokensCeiling,
  resolveExplicitCtxPin,
  resolveFitMaxSeqLength,
  retainedContextPin,
  unpinnedLoadContext,
  unreportedWindowMaxTokens,
} = await import("../src/features/chat/presets/preset-policy.ts");
const { deriveContextUsageBar } = await import(
  "../src/features/chat/lib/context-usage-bar-state.ts"
);

const GGUF = { is_gguf: true, context_length: 32768, max_context_length: 32768 };
const MLX = {
  is_gguf: false,
  is_mlx: true,
  context_length: 32768,
  native_context_length: 262144,
  max_context_length: 262144,
};

test("an MLX response carries a window without a native one", () => {
  // The whole point of the field split: before this, a non-GGUF response with no
  // native_context_length was discarded, which is every MLX model whose config
  // declares no trained window.
  assert.deepEqual(loadedContextFields({ is_gguf: false, is_mlx: true, context_length: 8192 }), {
    loadedContextLength: 8192,
    maxContextLength: 8192,
    nativeContextLength: null,
    loadedIsGguf: false,
    loadedContextEnforced: null,
  });
  // Transformers, which sizes nothing, still contributes no window.
  assert.deepEqual(loadedContextFields({ is_gguf: false, context_length: 2048 }), {
    loadedContextLength: null,
    maxContextLength: null,
    nativeContextLength: null,
    loadedIsGguf: false,
    loadedContextEnforced: null,
  });
  assert.equal(loadedContextFields(null).loadedIsGguf, null);
});

test("the enforcement verdict is a tri-state, and GGUF is true by construction", () => {
  assert.equal(loadedContextFields(GGUF).loadedContextEnforced, true);
  assert.equal(loadedContextFields(MLX).loadedContextEnforced, null);
  assert.equal(
    loadedContextFields({ ...MLX, context_length_enforced: true }).loadedContextEnforced,
    true,
  );
  assert.equal(
    loadedContextFields({ ...MLX, context_length_enforced: false }).loadedContextEnforced,
    false,
  );
});

test("a load that reported a non-GGUF backend outranks a stale variant", () => {
  assert.equal(isServedByLlamaCpp({ loadedIsGguf: true }), true);
  assert.equal(isServedByLlamaCpp({ activeGgufVariant: "Q4_K_M" }), true);
  // The variant and the path token outlive the pick that set them.
  assert.equal(
    isServedByLlamaCpp({ loadedIsGguf: false, activeGgufVariant: "Q4_K_M" }),
    false,
  );
  assert.equal(
    isServedByLlamaCpp({ loadedIsGguf: false, activeNativePathToken: "tok" }),
    false,
  );
  // A .gguf checkpoint before any load still reads as llama.cpp.
  assert.equal(isServedByLlamaCpp({ checkpoint: "/m/model.gguf" }), true);
  assert.equal(isServedByLlamaCpp({ checkpoint: "external::openai/gpt-4" }), false);
});

test("MLX is a Mac non-GGUF load, and the reasons that rule it out", () => {
  assert.equal(isServedByMlx(false, "mac", null), true);
  assert.equal(isServedByMlx(true, "mac", null), false);
  assert.equal(isServedByMlx(false, "cuda", null), false);
  for (const reason of ["mlx_unavailable", "intel_mac", "detection_failed"]) {
    assert.equal(isServedByMlx(false, "mac", reason), false, reason);
  }
});

test("an unpinned load asks a self-sizing backend for nothing", () => {
  assert.equal(unpinnedLoadContext(true, false, 4096), 0);
  assert.equal(unpinnedLoadContext(false, true, 4096), 0);
  assert.equal(unpinnedLoadContext(false, false, 4096), 4096);
  assert.equal(unpinnedLoadContext(false, null, DEFAULT_MAX_SEQ_LENGTH), 4096);
});

test("only MLX keeps its request as a pin after a load", () => {
  assert.equal(retainedContextPin({ isMlx: true, requestedContextLength: 32768 }), 32768);
  // Auto sends the sentinel, which is not a pin.
  assert.equal(retainedContextPin({ isMlx: true, requestedContextLength: 0 }), null);
  assert.equal(retainedContextPin({ isMlx: false, requestedContextLength: 32768 }), null);
  assert.equal(retainedContextPin({ requestedContextLength: 32768 }), null);
});

test("a preset records a window only where replaying it needs one", () => {
  assert.equal(capturedContextLength({ isGguf: true, controlPin: null, loadedContextLength: 32768 }), 32768);
  // Non-GGUF: an unpinned window is arrived at again on its own.
  assert.equal(capturedContextLength({ isGguf: false, controlPin: null, loadedContextLength: 32768 }), null);
  assert.equal(capturedContextLength({ isGguf: false, controlPin: 8192, loadedContextLength: 32768 }), 8192);
});

test("the reported window outranks the request it answered", () => {
  assert.equal(loadedContextForParams(32768, 0, 4096), 32768);
  assert.equal(loadedContextForParams(null, 8192, 4096), 8192);
  // The sentinel is below the control's minimum, so the previous value stands.
  assert.equal(loadedContextForParams(null, 0, 4096), 4096);
});

test("Max Tokens is bounded by the window, and never below its own minimum", () => {
  assert.equal(localMaxTokensCeiling(32768, 4096), 32768);
  assert.equal(localMaxTokensCeiling(null, 4096), 4096);
  // Reachable on MLX, which honours a tiny positive request verbatim. The control's
  // own minimum wins, since a slider whose maximum is below it cannot be operated.
  assert.equal(localMaxTokensCeiling(16, 4096), 64);
  assert.equal(unreportedWindowMaxTokens(true, 9000), 9000);
  assert.equal(unreportedWindowMaxTokens(false, 9000), DEFAULT_MAX_SEQ_LENGTH);
});

test("--fit owns sizing only for an unpinned GGUF on manual auto-layers", () => {
  assert.equal(resolveFitMaxSeqLength(true, "manual", -1, null, 4096), 0);
  assert.equal(resolveFitMaxSeqLength(true, "manual", -1, 8192, 4096), 8192);
  assert.equal(resolveFitMaxSeqLength(true, "manual", 20, null, 4096), 4096);
  assert.equal(resolveFitMaxSeqLength(true, "auto", -1, null, 4096), 4096);
  assert.equal(resolveFitMaxSeqLength(false, "manual", -1, null, 4096), 4096);
  assert.equal(resolveExplicitCtxPin(8192), 8192);
  assert.equal(resolveExplicitCtxPin(0), null);
  assert.equal(resolveExplicitCtxPin(null), null);
});

test("the usage bar names the three ways a window can end", () => {
  const at = (used: number, extra: Record<string, unknown> = {}) =>
    deriveContextUsageBar({
      used,
      total: 32768,
      isMlx: true,
      ...extra,
    })?.advice;
  assert.equal(at(1000), "none");
  assert.equal(at(30000), "mlx-near-limit");
  assert.equal(at(40000), "mlx-past-limit");
  assert.equal(at(30000, { isMlx: false }), "stops-at-limit");
  // A window the backend confirmed does not bound the cache is not a limit at all.
  assert.equal(at(30000, { contextEnforced: false }), "unenforced-limit");
  assert.equal(at(40000, { contextEnforced: false }), "unenforced-limit");
  assert.equal(at(30000, { contextEnforced: true }), "mlx-near-limit");
  assert.equal(at(30000, { contextEnforced: null }), "mlx-near-limit");
});
