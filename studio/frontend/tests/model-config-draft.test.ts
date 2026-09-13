// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  modelConfigDraftKey,
  patchModelConfigDraft,
  primeModelConfigDraft,
  readModelConfigDraft,
} from "../src/features/model-picker/model-config/model-config-draft.ts";
import type { PerModelConfig } from "../src/features/model-picker/model-config/per-model-config.ts";

const MODEL = "unsloth/Qwen3-8B-GGUF";
const VARIANT = "Q4_K_M";
const KEY = modelConfigDraftKey(MODEL, VARIANT);

const SEED: PerModelConfig = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: "f16",
  speculativeType: "auto",
  specDraftNMax: null,
  nParallel: null,
  nBatch: null,
  nUbatch: null,
  tensorParallel: false,
  disableVision: false,
  chatTemplateOverride: null,
  gpuMemoryMode: "auto",
  gpuLayers: -1,
  nCpuMoe: 0,
  selectedGpuIds: null,
};

test("two hosts share one draft for the same model identity", () => {
  primeModelConfigDraft(KEY, { config: SEED, remembered: true }, "none");
  patchModelConfigDraft(KEY, { kvCacheDtype: "q8_0" });
  const shared = readModelConfigDraft(KEY);
  assert.equal(shared?.config.kvCacheDtype, "q8_0");
  patchModelConfigDraft(KEY, { nParallel: 2 });
  assert.equal(readModelConfigDraft(KEY)?.config.nParallel, 2);
});

test("a new live signature re-seeds the draft from the resident process", () => {
  primeModelConfigDraft(KEY, { config: SEED, remembered: true }, "sig-a");
  patchModelConfigDraft(KEY, { kvCacheDtype: "q8_0" });
  const live: PerModelConfig = { ...SEED, kvCacheDtype: "q4_0", nParallel: 4 };
  primeModelConfigDraft(
    KEY,
    { config: live, remembered: true },
    "sig-b",
  );
  const reseeded = readModelConfigDraft(KEY);
  assert.equal(reseeded?.config.kvCacheDtype, "q4_0");
  assert.equal(reseeded?.config.nParallel, 4);
  assert.equal(reseeded?.appliedLiveSignature, "sig-b");
});
