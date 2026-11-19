// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The backend reads any positive max_seq_length as typed, so a resent resident window was never re-fitted.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const CONFIG_PAGE = readSrc("features/model-picker/components/model-config-page.tsx");
const policy = await import("../src/features/chat/presets/preset-policy.ts");

const MODEL = "unsloth/Qwen3.6-27B-MTP-GGUF";
const VARIANT = "UD-Q8_K_XL";

function sent({
  isGguf = true,
  customContextLength = null,
  gpuMemoryMode = "auto",
  gpuLayers = -1,
}: {
  isGguf?: boolean;
  customContextLength?: number | null;
  gpuMemoryMode?: "auto" | "manual";
  gpuLayers?: number;
} = {}) {
  return policy.resolveFitMaxSeqLength(
    isGguf,
    gpuMemoryMode,
    gpuLayers,
    customContextLength,
    policy.resolveLoadMaxSeqLength({
      modelId: MODEL,
      ggufVariant: isGguf ? VARIANT : null,
      isGguf,
      customContextLength,
      isMlx: false,
      pinnedMaxSeqLength: null,
      defaultMaxSeqLength: 4096,
    }),
  );
}

test("an Auto GGUF load sends 0 in every GPU memory mode", () => {
  assert.equal(sent(), 0);
  assert.equal(sent({ gpuMemoryMode: "manual" }), 0);
});

test("a typed context is sent as typed", () => {
  assert.equal(sent({ customContextLength: 65536 }), 65536);
  assert.equal(sent({ customContextLength: 65536, gpuMemoryMode: "manual" }), 65536);
});

test("non-GGUF loads keep the app default", () => {
  assert.equal(sent({ isGguf: false }), 4096);
});

test("no load path hands the resolver the resident window or the preset source", () => {
  for (const [label, source] of [
    ["runtime", RUNTIME],
    ["adapter", ADAPTER],
  ] as const) {
    for (const call of source.split("resolveLoadMaxSeqLength({").slice(1)) {
      const args = call.slice(0, call.indexOf("})"));
      assert.doesNotMatch(
        args,
        /\b(?:presetSource|loadedContextLength|loadedContextPin)\b/,
        label,
      );
    }
  }
  assert.doesNotMatch(RUNTIME, /max_seq_length_auto_derived/);
  assert.equal(
    (policy as Record<string, unknown>).isReplayedLoadContext,
    undefined,
  );
});

test("Manual with fixed layers still keeps the resident context on a same-model reload", () => {
  assert.match(RUNTIME, /loadCustomContextLength = loadContextLength;/);
});

test("an inherited Manual mode with fixed layers prices the context Reload keeps", async () => {
  const { fixedLayerContextPin, resolveEstimateContext } = await import(
    "../src/features/model-picker/model-config/estimate-context.ts"
  );
  const estimate = (
    config: { gpuMemoryMode?: "auto" | "manual"; gpuLayers: number; customContextLength: number | null },
    standingGpuMemoryMode: "auto" | "manual",
    activeLoadedContext: number | null,
  ) =>
    resolveEstimateContext(
      fixedLayerContextPin({
        isGguf: true,
        gpuMemoryMode: config.gpuMemoryMode,
        standingGpuMemoryMode,
        gpuLayers: config.gpuLayers,
        customContextLength: config.customContextLength,
        activeLoadedContext,
      }) ?? config.customContextLength,
    );
  const RESIDENT = 32768;
  // No per-model mode: performLoad applies the standing Manual preference and keeps the resident context.
  assert.equal(estimate({ gpuLayers: 20, customContextLength: null }, "manual", RESIDENT), RESIDENT);
  assert.equal(estimate({ gpuMemoryMode: "manual", gpuLayers: 20, customContextLength: null }, "auto", RESIDENT), RESIDENT);
  assert.equal(estimate({ gpuMemoryMode: "auto", gpuLayers: 20, customContextLength: null }, "manual", RESIDENT), 0);
  assert.equal(estimate({ gpuLayers: -1, customContextLength: null }, "manual", RESIDENT), 0);
  assert.equal(estimate({ gpuLayers: 20, customContextLength: 16384 }, "manual", RESIDENT), 16384);
  assert.equal(estimate({ gpuLayers: 20, customContextLength: null }, "manual", null), 0);
  assert.equal(
    CONFIG_PAGE.match(/standingGpuMemoryMode: gpuMemoryModeFallback,/g)?.length,
    2,
  );
  assert.match(
    RUNTIME,
    /const loadGpuMemoryMode =\s*pendingLoadConfig\?\.gpuMemoryMode \?\? stateBeforeUnload\.gpuMemoryMode;/,
  );
});

test("the memory estimate prices the fit for every Auto GGUF load", () => {
  assert.doesNotMatch(CONFIG_PAGE, /activePresetSource/);
  assert.match(
    CONFIG_PAGE,
    /resolveEstimateContext\(runtimeConfig\.customContextLength \?\? null\)/,
  );
});
