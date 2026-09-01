// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Behaviour of the pure helpers the MLX context work introduced. The rest of their
// coverage regexes the source, which passes with the body deleted; these call them.

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
  residentIsServedByMlx,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const {
  capturedContextLength,
  loadedContextForParams,
  resolveLoadMaxSeqLength,
  loadRequestContextPin,
  localMaxTokensCeiling,
  replayMaxTokensCap,
  resolveExplicitCtxPin,
  resolveFitMaxSeqLength,
  retainedContextPin,
  unpinnedDefaultRequest,
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
  // The point of the field split: a non-GGUF response with no native_context_length used
  // to be discarded, which is every MLX model declaring no trained window.
  assert.deepEqual(loadedContextFields({ is_gguf: false, is_mlx: true, context_length: 8192 }), {
    loadedContextLength: 8192,
    maxContextLength: 8192,
    nativeContextLength: null,
    loadedIsGguf: false,
    loadedIsMlx: true,
    loadedContextEnforced: null,
  });
  // Transformers, which sizes nothing, still contributes no window.
  assert.deepEqual(loadedContextFields({ is_gguf: false, context_length: 2048 }), {
    loadedContextLength: null,
    maxContextLength: null,
    nativeContextLength: null,
    loadedIsGguf: false,
    loadedIsMlx: null,
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
  // Reachable on MLX, which honours a tiny request verbatim; the control's minimum wins,
  // since a slider whose maximum is below it cannot be operated.
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
  // Bounded unless a case says otherwise: an unconfirmed window is its own answer below.
  const at = (used: number, extra: Record<string, unknown> = {}) =>
    deriveContextUsageBar({
      used,
      total: 32768,
      isMlx: true,
      contextEnforced: true,
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
  // Unjudged says the same thing operationally: a probe that could not build a cache
  // installed no window, so it grows exactly as a confirmed unenforced one does, and
  // rotating-cache advice would be the opposite of true.
  assert.equal(at(30000, { contextEnforced: null }), "unenforced-limit");
  assert.equal(at(30000, { contextEnforced: undefined }), "unenforced-limit");
  // Only for MLX: nothing else installs a window whose bound could go unjudged.
  assert.equal(at(30000, { isMlx: false, contextEnforced: null }), "stops-at-limit");
});

test("an outgoing self-sizing window does not become the next model's request", () => {
  // An unpinned MLX load writes its RESOLVED window into params.maxSeqLength.
  const afterMlx = loadedContextForParams(131072, 0, 4096);
  assert.equal(afterMlx, 131072);
  // An unconfigured transformers model still asks for the app default: 131072 there is
  // an allocation nobody requested.
  assert.equal(unpinnedDefaultRequest(true, afterMlx, DEFAULT_MAX_SEQ_LENGTH), 4096);
  // A backend that does not size its own window leaves the session request intact.
  assert.equal(unpinnedDefaultRequest(false, 8192, DEFAULT_MAX_SEQ_LENGTH), 8192);
  assert.equal(unpinnedDefaultRequest(false, 0, DEFAULT_MAX_SEQ_LENGTH), 4096);
  assert.equal(unpinnedDefaultRequest(null, null, DEFAULT_MAX_SEQ_LENGTH), 4096);
  // End to end: the resolver must not hand the outgoing window to the new load.
  assert.equal(
    resolveLoadMaxSeqLength({
      modelId: "org/plain-transformers",
      ggufVariant: null,
      isGguf: false,
      customContextLength: null,
      loadedContextLength: null,
      currentCheckpoint: "mlx-community/Some-MLX",
      activeGgufVariant: null,
      isMlx: false,
      pinnedMaxSeqLength: null,
      defaultMaxSeqLength: unpinnedDefaultRequest(true, afterMlx, DEFAULT_MAX_SEQ_LENGTH),
      presetSource: "builtin-default",
    }),
    4096,
  );
});

test("the backend's own is_mlx vetoes the platform for a resident model", () => {
  const MAC = ["mac", null] as const;
  // A native-audio checkpoint loads on Apple Silicon through NativeAudioBackend, which
  // the worker picks before the MLX fast-path, so its settings are not MLX's to clear.
  assert.equal(residentIsServedByMlx(false, ...MAC, false), false);
  assert.equal(residentIsServedByMlx(false, ...MAC, true), true);
  // Nothing loaded yet: the platform is still the best answer available.
  assert.equal(residentIsServedByMlx(false, ...MAC, null), true);
  assert.equal(residentIsServedByMlx(false, ...MAC, undefined), true);
  // The veto adds to the platform rule rather than replacing it.
  assert.equal(residentIsServedByMlx(true, ...MAC, true), false);
  assert.equal(residentIsServedByMlx(false, "linux", null, true), false);
  // And the response carries that answer, so the store never has to infer it.
  assert.equal(loadedContextFields({ is_gguf: false, is_mlx: true, context_length: 8192 }).loadedIsMlx, true);
  assert.equal(
    loadedContextFields({ is_gguf: false, is_mlx: false, context_length: 2048 }).loadedIsMlx,
    false,
  );
  // Omitted is unknown, not a denial: an older backend answers nothing here.
  assert.equal(loadedContextFields({ is_gguf: false, context_length: 2048 }).loadedIsMlx, null);
  assert.equal(loadedContextFields({ is_gguf: true, context_length: 4096 }).loadedIsMlx, null);
  assert.equal(loadedContextFields(null).loadedIsMlx, null);
});

test("a cap never lands below the Max Tokens control's own minimum", () => {
  // MLX honours a tiny positive request verbatim, and the cap only lowers Max Tokens, so
  // a raw window here would clamp the value outside its slider.
  assert.equal(replayMaxTokensCap(32), 64);
  assert.equal(replayMaxTokensCap(64), 64);
  assert.equal(replayMaxTokensCap(32768), 32768);
  // Nothing sized a window: no cap at all, which is not a cap of zero.
  assert.equal(replayMaxTokensCap(null), undefined);
  assert.equal(replayMaxTokensCap(undefined), undefined);
  // The same floor the displayed ceiling already used, so the pair cannot disagree.
  assert.equal(replayMaxTokensCap(32), localMaxTokensCeiling(32, 32));
});

test("a load keeps the pin it was built from, wherever the record held it", () => {
  // resolveLoadMaxSeqLength takes the pre-move field for an unpinned MLX target, so the
  // load must pin the same number or the UI shows Auto for a pinned runtime.
  assert.equal(loadRequestContextPin(null, true, 8192), 8192);
  assert.equal(loadRequestContextPin(32768, true, 8192), 32768, "the live field leads");
  // llama.cpp's maxSeqLength is not a context pin, so only MLX admits it.
  assert.equal(loadRequestContextPin(null, false, 8192), null);
  assert.equal(loadRequestContextPin(null, true, null), null);
  // The request the two agree on: same input, same number.
  assert.equal(
    loadRequestContextPin(null, true, 8192),
    resolveLoadMaxSeqLength({
      modelId: "org/mlx-model",
      isGguf: false,
      customContextLength: null,
      loadedContextLength: null,
      currentCheckpoint: "",
      isMlx: true,
      pinnedMaxSeqLength: 8192,
      defaultMaxSeqLength: DEFAULT_MAX_SEQ_LENGTH,
      presetSource: "custom",
    }),
  );
});
