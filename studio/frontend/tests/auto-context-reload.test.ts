// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The backend reads any positive max_seq_length as typed, so a resent resident window was never re-fitted.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const RUNTIME = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
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

test("Manual with fixed layers keeps the resident context only on a same-model Auto reload", async () => {
  const { fixedLayerContextPin, resolveEstimateContext } = await import(
    "../src/features/model-picker/model-config/estimate-context.ts"
  );
  const RESIDENT = 32768;
  const pin = ({
    isGguf = true,
    gpuMemoryMode = "manual",
    gpuLayers = 20,
    customContextLength = null,
    activeLoadedContext = RESIDENT,
  }: {
    isGguf?: boolean;
    gpuMemoryMode?: "auto" | "manual";
    gpuLayers?: number | null;
    customContextLength?: number | null;
    activeLoadedContext?: number | null;
  } = {}) =>
    fixedLayerContextPin({ isGguf, gpuMemoryMode, gpuLayers, customContextLength, activeLoadedContext });
  assert.equal(pin(), RESIDENT);
  assert.equal(pin({ gpuLayers: 0 }), RESIDENT);
  assert.equal(pin({ gpuMemoryMode: "auto" }), null);
  assert.equal(pin({ gpuLayers: -1 }), null);
  assert.equal(pin({ gpuLayers: null }), null);
  assert.equal(pin({ customContextLength: 16384 }), null);
  assert.equal(pin({ isGguf: false }), null);
  assert.equal(pin({ activeLoadedContext: null }), null);
  assert.equal(pin({ activeLoadedContext: 0 }), null);
  assert.equal(resolveEstimateContext(pin() ?? null), RESIDENT);

  // Both the load and the config page resolve an unset per-model mode before asking.
  assert.match(
    RUNTIME,
    /loadCustomContextLength =\s*fixedLayerContextPin\(\{\s*isGguf: isGguf && !switchingModelOrVariant,\s*gpuMemoryMode: loadGpuMemoryMode,\s*gpuLayers: loadGpuLayers,\s*customContextLength: loadCustomContextLength,\s*activeLoadedContext: loadContextLength,\s*\}\) \?\? loadCustomContextLength;/,
  );
  assert.match(
    RUNTIME,
    /const loadGpuMemoryMode =\s*pendingLoadConfig\?\.gpuMemoryMode \?\? stateBeforeUnload\.gpuMemoryMode;/,
  );
  assert.equal(
    CONFIG_PAGE.match(/gpuMemoryMode: (?:loadableConfig|effectiveConfig)\.gpuMemoryMode \?\? gpuMemoryModeFallback,/g)?.length,
    2,
  );
});

test("the memory estimate prices the fit for every Auto GGUF load", () => {
  assert.match(
    CONFIG_PAGE,
    /resolveEstimateContext\(runtimeConfig\.customContextLength \?\? null\)/,
  );
});
