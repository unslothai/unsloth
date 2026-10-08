// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const { settingIsDefault, settingResetPatch } = await import(
  "../src/features/model-picker/model-config/setting-reset.ts"
);
const { DEFAULT_PER_MODEL_CONFIG, isDefaultConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

type Config = typeof DEFAULT_PER_MODEL_CONFIG;
type Setting = Parameters<typeof settingIsDefault>[1];

const CHANGED: [Setting, Partial<Config>][] = [
  ["contextLength", { customContextLength: 34432 }],
  ["contextPin", { maxSeqLength: 8192 }],
  ["kvCacheDtype", { kvCacheDtype: "q8_0" }],
  ["mlxKvQuant", { mlxKvQuant: "8" as Config["mlxKvQuant"] }],
  ["mlxInt8Prefill", { mlxInt8Prefill: true }],
  ["speculative", { speculativeType: "mtp", specDraftNMax: 3 }],
  ["specDraftNMax", { specDraftNMax: 3 }],
  ["specDraftCacheDtype", { specDraftCacheDtype: "q8_0" }],
  ["nParallel", { nParallel: 4 }],
  ["nBatch", { nBatch: 4096 }],
  ["nUbatch", { nUbatch: 1024 }],
  ["tensorParallel", { tensorParallel: true }],
  ["vision", { disableVision: true }],
  ["reasoningBudget", { reasoningBudget: 0 }],
  ["reasoningBudgetMessage", { reasoningBudgetMessage: "Done thinking." }],
  ["gpuMemory", { gpuMemoryMode: "manual", gpuLayers: 20, nCpuMoe: 4 }],
  ["loadMode", { loadMode: "mlock" }],
  ["ctxCheckpoints", { ctxCheckpoints: 0 }],
  ["cacheRam", { cacheRam: -1 }],
];

test("the defaults read as default for every setting", () => {
  for (const [setting] of CHANGED) {
    assert.equal(
      settingIsDefault(DEFAULT_PER_MODEL_CONFIG, setting),
      true,
      setting,
    );
  }
});

test("each setting's reset brings only that setting back to default", () => {
  for (const [setting, change] of CHANGED) {
    const changed = { ...DEFAULT_PER_MODEL_CONFIG, ...change };
    assert.equal(settingIsDefault(changed, setting), false, setting);
    const reset = { ...changed, ...settingResetPatch(setting) };
    assert.equal(settingIsDefault(reset, setting), true, setting);
    // Same verdict storage uses, so a reset field never keeps an entry alive on its own.
    assert.equal(isDefaultConfig(reset), true, setting);
  }
});

test("a field reset leaves the other fields alone", () => {
  const changed = {
    ...DEFAULT_PER_MODEL_CONFIG,
    kvCacheDtype: "q8_0",
    nParallel: 4,
  };
  const reset = { ...changed, ...settingResetPatch("kvCacheDtype") };
  assert.equal(reset.kvCacheDtype, null);
  assert.equal(reset.nParallel, 4);
});

test("a patch is a copy, so a caller cannot edit the table", () => {
  const patch = settingResetPatch("nParallel");
  patch.nParallel = 8;
  assert.equal(settingResetPatch("nParallel").nParallel, null);
});

test("Auto speculative decoding reads as the default, as the select and a loaded model write it", () => {
  for (const value of [null, "auto", "Auto", "default"]) {
    assert.equal(
      settingIsDefault(
        { ...DEFAULT_PER_MODEL_CONFIG, speculativeType: value },
        "speculative",
      ),
      true,
      `speculativeType=${value}`,
    );
  }
  assert.equal(
    settingIsDefault(
      { ...DEFAULT_PER_MODEL_CONFIG, speculativeType: "ngram" },
      "speculative",
    ),
    false,
  );
});

test("a non-GGUF context reset clears a pin in either field", () => {
  const both = {
    ...DEFAULT_PER_MODEL_CONFIG,
    customContextLength: 16384,
    maxSeqLength: 8192,
  };
  assert.equal(settingIsDefault(both, "contextPin"), false);
  const reset = { ...both, ...settingResetPatch("contextPin") };
  assert.equal(reset.customContextLength, null);
  assert.equal(reset.maxSeqLength, null);
});
