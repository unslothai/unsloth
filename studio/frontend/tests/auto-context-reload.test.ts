// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A same-model GGUF reload on Auto asks the server to fit the context again, whatever preset is
// active. Resending the resident window made the backend treat it as a typed context, so a reload
// after a KV cache, vision or GPU change kept the old window under any preset but a pristine Default.

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
  // --fit off would launch at the native context otherwise; that path pins it explicitly.
  assert.match(RUNTIME, /loadCustomContextLength = loadContextLength;/);
});

test("the memory estimate prices the fit for every Auto GGUF load", () => {
  assert.doesNotMatch(CONFIG_PAGE, /activePresetSource/);
  assert.match(
    CONFIG_PAGE,
    /resolveEstimateContext\(\s*runtimeConfig\.customContextLength \?\? null,\s*activeLoadedContext,\s*(?:\/\/[^\n]*\n\s*)?target\.isGguf === true,\s*\)/,
  );
});
