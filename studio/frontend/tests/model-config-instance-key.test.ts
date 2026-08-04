// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { modelConfigInstanceKey } from "../src/features/model-picker/model-config/config-signature.ts";
import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";

const MODEL = "unsloth/Qwen3-8B-GGUF";
const VARIANT = "Q4_K_M";

// What the model is actually running with, as useActiveModelConfig reports it.
const LIVE: PerModelConfig = {
  customContextLength: 16384,
  maxSeqLength: null,
  kvCacheDtype: "q8_0",
  speculativeType: "ngram",
  specDraftNMax: 6,
  nParallel: 4,
  tensorParallel: true,
  chatTemplateOverride: null,
  gpuMemoryMode: "manual",
  gpuLayers: 24,
  nCpuMoe: 3,
  selectedGpuIds: [0, 1],
};

// What ModelConfigPage would fall back to before the live config lands.
const SAVED: PerModelConfig = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  speculativeType: "auto",
  specDraftNMax: null,
  nParallel: null,
  tensorParallel: false,
  chatTemplateOverride: null,
  gpuMemoryMode: "auto",
  gpuLayers: -1,
  nCpuMoe: 0,
  selectedGpuIds: null,
};

/**
 * ModelConfigPage reads `loadedConfig` in a useState initializer, so it seeds once per
 * MOUNTED instance, and React keeps that instance while the key is unchanged.
 */
function renderEditor(
  previous: { key: string; editing: PerModelConfig } | null,
  key: string,
  loadedConfig: PerModelConfig | null,
): { key: string; editing: PerModelConfig } {
  if (previous && previous.key === key) {
    return previous;
  }
  return { key, editing: loadedConfig ?? SAVED };
}

test("the settings editor re-seeds when the live config arrives after mount", () => {
  // Opened before status answered: loadedConfig is null first and live on the next render.
  let editor = renderEditor(
    null,
    modelConfigInstanceKey(MODEL, VARIANT, null),
    null,
  );
  assert.deepEqual(editor.editing, SAVED);

  editor = renderEditor(
    editor,
    modelConfigInstanceKey(MODEL, VARIANT, LIVE),
    LIVE,
  );
  // Without the live config in the key the editor would still hold SAVED, and Apply
  // would reload the model over what it is running with.
  assert.deepEqual(editor.editing, LIVE);
});

test("a repeated status poll keeps the same editor instance", () => {
  const first = renderEditor(
    null,
    modelConfigInstanceKey(MODEL, VARIANT, LIVE),
    LIVE,
  );
  // An equal config from the next poll must not remount and discard what was typed.
  const again = renderEditor(
    first,
    modelConfigInstanceKey(MODEL, VARIANT, { ...LIVE }),
    LIVE,
  );
  assert.equal(again, first);
});

test("every mirrored setting moves the instance key", () => {
  const base = modelConfigInstanceKey(MODEL, VARIANT, LIVE);
  const changes: PerModelConfig[] = [
    { ...LIVE, customContextLength: 8192 },
    { ...LIVE, maxSeqLength: 4096 },
    { ...LIVE, kvCacheDtype: "f16" },
    { ...LIVE, speculativeType: "off" },
    { ...LIVE, specDraftNMax: 4 },
    { ...LIVE, nParallel: 1 },
    { ...LIVE, tensorParallel: false },
    { ...LIVE, chatTemplateOverride: "{{ bos_token }}" },
    { ...LIVE, gpuMemoryMode: "auto" },
    { ...LIVE, gpuLayers: 20 },
    { ...LIVE, nCpuMoe: 0 },
    { ...LIVE, selectedGpuIds: [0] },
  ];
  for (const changed of changes) {
    assert.notEqual(modelConfigInstanceKey(MODEL, VARIANT, changed), base);
  }
  // The GPU pick is a set, not an order.
  assert.equal(
    modelConfigInstanceKey(MODEL, VARIANT, { ...LIVE, selectedGpuIds: [1, 0] }),
    base,
  );
});

test("the model and its quant still key the editor", () => {
  const base = modelConfigInstanceKey(MODEL, VARIANT, LIVE);
  assert.notEqual(modelConfigInstanceKey("unsloth/Other-GGUF", VARIANT, LIVE), base);
  assert.notEqual(modelConfigInstanceKey(MODEL, "Q8_0", LIVE), base);
  // A loose .gguf carries no quant; null and undefined are the same absence.
  assert.equal(
    modelConfigInstanceKey(MODEL, null, LIVE),
    modelConfigInstanceKey(MODEL, undefined, LIVE),
  );
});
