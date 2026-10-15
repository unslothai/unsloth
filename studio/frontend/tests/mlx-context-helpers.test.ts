// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
  resumesThought,
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
  // A non-GGUF response with no native_context_length used to be discarded.
  assert.deepEqual(loadedContextFields({ is_gguf: false, is_mlx: true, context_length: 8192 }), {
    loadedContextLength: 8192,
    maxContextLength: 8192,
    nativeContextLength: null,
    loadedIsGguf: false,
    loadedIsMlx: true,
    loadedContextEnforced: null,
    loadedContextUnboundedWhenBatched: false,
    loadedParallelSlots: null,
    loadedContextBudget: null,
  });
  assert.deepEqual(loadedContextFields({ is_gguf: false, context_length: 2048 }), {
    loadedContextLength: null,
    maxContextLength: null,
    nativeContextLength: null,
    loadedIsGguf: false,
    loadedIsMlx: null,
    loadedContextEnforced: null,
    loadedContextUnboundedWhenBatched: false,
    loadedParallelSlots: null,
    loadedContextBudget: null,
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

test("a limit kept as a per-request budget travels beside the enforcement verdict", () => {
  const budgeted = loadedContextFields({
    ...MLX,
    context_length_enforced: false,
    mlx_context_budget: 32768,
  });
  assert.equal(budgeted.loadedContextBudget, 32768);
  assert.equal(budgeted.loadedContextEnforced, false);
  assert.equal(loadedContextFields(MLX).loadedContextBudget, null);
  assert.equal(loadedContextFields({ ...GGUF, mlx_context_budget: 8192 }).loadedContextBudget, null);
});

test("a budgeted limit is advised on as a refusal, not as a slowdown or a decoration", () => {
  const advice = (extra: Record<string, unknown>) =>
    deriveContextUsageBar({ used: 30000, total: 32768, isMlx: true, ...extra })?.advice;

  assert.equal(advice({ contextEnforced: false, contextBudget: 32768 }), "mlx-refuses-past-limit");
  assert.equal(advice({ contextEnforced: false }), "unenforced-limit");
  assert.equal(advice({ contextEnforced: true }), "mlx-near-limit");
  assert.equal(
    deriveContextUsageBar({ used: 100, total: 32768, isMlx: true, contextBudget: 32768 })?.advice,
    "none",
  );
});

test("a load that reported a non-GGUF backend outranks a stale variant", () => {
  assert.equal(isServedByLlamaCpp({ loadedIsGguf: true }), true);
  assert.equal(isServedByLlamaCpp({ activeGgufVariant: "Q4_K_M" }), true);
  assert.equal(
    isServedByLlamaCpp({ loadedIsGguf: false, activeGgufVariant: "Q4_K_M" }),
    false,
  );
  assert.equal(
    isServedByLlamaCpp({ loadedIsGguf: false, activeNativePathToken: "tok" }),
    false,
  );
  assert.equal(isServedByLlamaCpp({ checkpoint: "/m/model.gguf" }), true);
  assert.equal(isServedByLlamaCpp({ checkpoint: "external::openai/gpt-4" }), false);
});

test("a thought resumes on llama-server and on a load MLX reports serving", () => {
  assert.equal(resumesThought({ loadedIsGguf: true }), true);
  assert.equal(resumesThought({ loadedIsGguf: false, loadedIsMlx: true }), true);
  assert.equal(resumesThought({ loadedIsGguf: false, loadedIsMlx: false }), false);
  assert.equal(resumesThought({ loadedIsMlx: null }), false);
  assert.equal(
    resumesThought({ loadedIsMlx: true, checkpoint: "external::openai/gpt-4" }),
    false,
  );
});

test("MLX is a Mac non-GGUF load, and the reasons that rule it out", () => {
  assert.equal(isServedByMlx(false, "mac", null), true);
  assert.equal(isServedByMlx(true, "mac", null), false);
  assert.equal(isServedByMlx(false, "cuda", null), false);
  for (const reason of [
    "mlx_unavailable",
    "no_torch",
    "intel_mac",
    "detection_failed",
  ]) {
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
  // MLX honours a tiny request verbatim; the control's minimum wins.
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
      contextEnforced: true,
      ...extra,
    })?.advice;
  assert.equal(at(1000), "none");
  assert.equal(at(30000), "mlx-near-limit");
  assert.equal(at(40000), "mlx-past-limit");
  assert.equal(at(30000, { isMlx: false }), "stops-at-limit");
  assert.equal(at(30000, { contextEnforced: false }), "unenforced-limit");
  assert.equal(at(40000, { contextEnforced: false }), "unenforced-limit");
  assert.equal(at(30000, { contextEnforced: true }), "mlx-near-limit");
  // Unjudged means no window was installed, so it grows like an unenforced one.
  assert.equal(at(30000, { contextEnforced: null }), "unenforced-limit");
  assert.equal(at(30000, { contextEnforced: undefined }), "unenforced-limit");
  assert.equal(at(30000, { isMlx: false, contextEnforced: null }), "stops-at-limit");
  const batched = { contextUnboundedWhenBatched: true };
  assert.equal(at(30000, { ...batched, parallelSlots: 4 }), "unenforced-limit");
  assert.equal(at(40000, { ...batched, parallelSlots: 4 }), "unenforced-limit");
  assert.equal(at(30000, { ...batched, parallelSlots: 1 }), "mlx-near-limit");
  assert.equal(at(30000, { ...batched, parallelSlots: null }), "mlx-near-limit");
  assert.equal(at(30000, { parallelSlots: 4 }), "mlx-near-limit");
});

test("an outgoing self-sizing window does not become the next model's request", () => {
  const afterMlx = loadedContextForParams(131072, 0, 4096);
  assert.equal(afterMlx, 131072);
  assert.equal(unpinnedDefaultRequest(true, afterMlx, DEFAULT_MAX_SEQ_LENGTH), 4096);
  assert.equal(unpinnedDefaultRequest(false, 8192, DEFAULT_MAX_SEQ_LENGTH), 8192);
  assert.equal(unpinnedDefaultRequest(false, 0, DEFAULT_MAX_SEQ_LENGTH), 4096);
  assert.equal(unpinnedDefaultRequest(null, null, DEFAULT_MAX_SEQ_LENGTH), 4096);
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
  // Native-audio loads pick NativeAudioBackend before the MLX fast path.
  assert.equal(residentIsServedByMlx(false, ...MAC, false), false);
  assert.equal(residentIsServedByMlx(false, ...MAC, true), true);
  assert.equal(residentIsServedByMlx(false, ...MAC, null), true);
  assert.equal(residentIsServedByMlx(false, ...MAC, undefined), true);
  assert.equal(residentIsServedByMlx(true, ...MAC, true), false);
  assert.equal(residentIsServedByMlx(false, "linux", null, true), false);
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
  // MLX honours a tiny request verbatim, so a raw window would clamp outside the slider.
  assert.equal(replayMaxTokensCap(32), 64);
  assert.equal(replayMaxTokensCap(64), 64);
  assert.equal(replayMaxTokensCap(32768), 32768);
  assert.equal(replayMaxTokensCap(null), undefined);
  assert.equal(replayMaxTokensCap(undefined), undefined);
  assert.equal(replayMaxTokensCap(32), localMaxTokensCeiling(32, 32));
});

test("a load keeps the pin it was built from, wherever the record held it", () => {
  // resolveLoadMaxSeqLength reads the pre-move field for unpinned MLX, so pin the same number.
  assert.equal(loadRequestContextPin(null, true, 8192), 8192);
  assert.equal(loadRequestContextPin(32768, true, 8192), 32768, "the live field leads");
  // llama.cpp's maxSeqLength is not a context pin, so only MLX admits it.
  assert.equal(loadRequestContextPin(null, false, 8192), null);
  assert.equal(loadRequestContextPin(null, true, null), null);
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

test("a load response's two window facts and width reach the bar together", () => {
  const wide = loadedContextFields({
    is_mlx: true,
    context_length: 32768,
    native_context_length: 32768,
    context_length_enforced: true,
    context_unbounded_when_batched: true,
    parallel_slots: 4,
  });
  assert.equal(wide.loadedContextUnboundedWhenBatched, true);
  assert.equal(wide.loadedParallelSlots, 4);

  assert.equal(
    deriveContextUsageBar({
      used: 30000,
      total: 32768,
      isMlx: true,
      contextEnforced: wide.loadedContextEnforced,
      contextUnboundedWhenBatched: wide.loadedContextUnboundedWhenBatched,
      parallelSlots: wide.loadedParallelSlots,
    })?.advice,
    "unenforced-limit",
  );

  const gguf = loadedContextFields({
    is_gguf: true,
    context_length: 32768,
    native_context_length: 32768,
    context_unbounded_when_batched: true,
    parallel_slots: 4,
  });
  assert.equal(gguf.loadedContextUnboundedWhenBatched, false);
});

test("a background load cannot leave the visible model reading another model's window", () => {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  const list = source.slice(
    source.indexOf("const VISIBLE_MODEL_RUNTIME_KEYS = ["),
    source.indexOf("] as const satisfies"),
  );
  for (const key of [
    "loadedContextEnforced",
    "loadedContextUnboundedWhenBatched",
    "loadedParallelSlots",
  ]) {
    assert.match(list, new RegExp(`"${key}"`), `${key} is not preserved`);
  }
});

test("a queued run keeps its own model's MLX thought-resume verdict", async () => {
  const { snapshotQueuedChatRunSettings } = await import(
    "../src/features/chat/utils/queued-chat-run-settings.ts"
  );
  const resident = {
    params: { checkpoint: "mlx-community/Qwen3-0.6B-4bit" },
    activeGgufVariant: null,
    activeNativePathToken: null,
    loadedIsGguf: false,
    loadedIsMlx: true,
  };
  const queued = snapshotQueuedChatRunSettings(
    resident as unknown as Parameters<typeof snapshotQueuedChatRunSettings>[0],
  );
  const live = { ...resident, ...loadedContextFields(null) };
  const runtime = { ...live, ...queued };
  assert.equal(
    resumesThought({
      loadedIsGguf: runtime.loadedIsGguf,
      loadedIsMlx: runtime.loadedIsMlx,
      activeGgufVariant: runtime.activeGgufVariant,
      activeNativePathToken: runtime.activeNativePathToken,
      checkpoint: runtime.params.checkpoint,
    }),
    true,
  );
});
