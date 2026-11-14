// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import * as gpuTensorSplit from "../src/hooks/gpu-tensor-split.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

registerBundlerResolver();
const {
  DEFAULT_PER_MODEL_CONFIG,
  normalizePerModelConfig,
  sameEngramPrecision,
  storedSpeculativeAuto,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
// Loaded as the other config tests load it: its app imports stubbed, its comparison real.
const { perModelConfigsEqual } = loadWithStubs<{
  perModelConfigsEqual: (a: unknown, b: unknown) => boolean;
}>(
  new URL(
    "../src/features/model-picker/model-config/apply-per-model-config.ts",
    import.meta.url,
  ),
  {
    "@/features/chat/stores/chat-runtime-store": {
      normalizeSpeculativeType: (value: string | null | undefined) =>
        value == null || value === "" ? null : value.toLowerCase(),
      readPersistedSpeculativeType: () => "auto",
    },
    "@/features/chat/presets/preset-policy": {},
    "@/hooks/gpu-tensor-split": gpuTensorSplit,
    "./config-signature": { gpuFieldsSignature: () => "" },
    "./per-model-config": {
      normalizeMaxSeqLength: (value: number | null | undefined) =>
        value ?? null,
      sameEngramPrecision,
    },
  },
);

// A model loaded on its defaults reports them spelled out, as the runtime resolved them.
const loadedOnDefaults = {
  ...DEFAULT_PER_MODEL_CONFIG,
  speculativeType: "auto",
  gpuLayers: -1,
  nCpuMoe: 0,
  llamaExtraArgs: [],
} as typeof DEFAULT_PER_MODEL_CONFIG;

const asStored = (config: typeof DEFAULT_PER_MODEL_CONFIG) =>
  normalizePerModelConfig(storedSpeculativeAuto(config, false));

test("a model loaded on its defaults counts as Default, not an edit", () => {
  // Read raw, these differ: this is what kept a pinned model at Unsaved changes after every Save.
  assert.equal(
    perModelConfigsEqual(loadedOnDefaults, DEFAULT_PER_MODEL_CONFIG),
    false,
  );
  assert.equal(
    perModelConfigsEqual(
      asStored(loadedOnDefaults),
      asStored(DEFAULT_PER_MODEL_CONFIG),
    ),
    true,
  );
});

test("the page judges Default the way the set store compares settings", () => {
  const page = readSrc(
    "features/model-picker/components/model-config-page.tsx",
  );
  const at = page.indexOf("const atDefault =");
  assert.notEqual(at, -1);
  const body = page.slice(at, at + 400);
  assert.match(
    body,
    /normalizePerModelConfig\(\s*storedSpeculativeAuto\(\{ \.\.\.config, customContextLength: null \}, targetIsMlx\)/,
  );
  assert.match(
    body,
    /storedSpeculativeAuto\(DEFAULT_PER_MODEL_CONFIG, targetIsMlx\)/,
  );
});
