// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const {
  adoptLegacyConfigKey,
  listPerModelConfigs,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

// Older releases keyed repos cached outside the active HF cache by snapshot path.
const LEGACY_ID = "/home/u/.cache/models/snapshots/2f1c9ab";
const MODEL_ID = "unsloth/Repo-GGUF";

function config(
  maxSeqLength: number,
  kvCacheDtype: string | null = null,
  chatTemplateOverride: string | null = null,
) {
  return {
    customContextLength: null,
    maxSeqLength,
    kvCacheDtype,
    speculativeType: null,
    specDraftNMax: null,
    nParallel: null,
    reasoningBudget: -1,
    reasoningBudgetMessage: "",
    nBatch: null,
    nUbatch: null,
    tensorParallel: false,
    disableVision: false,
    chatTemplateOverride,
  };
}

// Mirrors MAX_ENTRIES in per-model-config.ts, which does not export it.
const MAX_ENTRIES = 500;
// Large templates hit MAX_PER_MODEL_CONFIG_STORAGE_BYTES (1 MiB) before the entry budget.
const BIG_TEMPLATE = "x".repeat(60_000);
const TEMPLATE_MODELS = 16;

// Assert values, not just the key: a wrong-argument save writes all defaults and still succeeds.
test("a legacy-keyed config moves to the current id with its values", () => {
  store.clear();
  savePerModelConfig(LEGACY_ID, "Q4_K_M", config(32768, "q8_0"));

  assert.equal(adoptLegacyConfigKey(MODEL_ID, LEGACY_ID, "Q4_K_M"), true);

  const adopted = resolveInitialConfig(MODEL_ID, "Q4_K_M");
  assert.equal(adopted.remembered, true);
  assert.equal(adopted.config.maxSeqLength, 32768);
  assert.equal(adopted.config.kvCacheDtype, "q8_0");
  assert.equal(listPerModelConfigs().length, 1);
  assert.equal(resolveInitialConfig(LEGACY_ID, "Q4_K_M").remembered, false);
});

test("a config already saved under the current id wins and the stale one still goes", () => {
  store.clear();
  savePerModelConfig(LEGACY_ID, "Q4_K_M", config(4096));
  savePerModelConfig(MODEL_ID, "Q4_K_M", config(131072, "q4_0"));

  assert.equal(adoptLegacyConfigKey(MODEL_ID, LEGACY_ID, "Q4_K_M"), true);

  const kept = resolveInitialConfig(MODEL_ID, "Q4_K_M");
  assert.equal(kept.config.maxSeqLength, 131072);
  assert.equal(kept.config.kvCacheDtype, "q4_0");
  assert.equal(listPerModelConfigs().length, 1);
  assert.equal(resolveInitialConfig(LEGACY_ID, "Q4_K_M").remembered, false);
});

test("adopting one quant leaves another quant of the same model alone", () => {
  store.clear();
  savePerModelConfig(LEGACY_ID, "Q4_K_M", config(32768));
  savePerModelConfig(LEGACY_ID, "Q8_0", config(8192));

  assert.equal(adoptLegacyConfigKey(MODEL_ID, LEGACY_ID, "Q4_K_M"), true);

  assert.equal(
    resolveInitialConfig(MODEL_ID, "Q4_K_M").config.maxSeqLength,
    32768,
  );
  assert.equal(
    resolveInitialConfig(LEGACY_ID, "Q8_0").config.maxSeqLength,
    8192,
  );
  assert.equal(resolveInitialConfig(MODEL_ID, "Q8_0").remembered, false);
  assert.equal(listPerModelConfigs().length, 2);
});

test("nothing to move is not a move", () => {
  store.clear();
  assert.equal(adoptLegacyConfigKey(MODEL_ID, LEGACY_ID, "Q4_K_M"), false);

  savePerModelConfig(MODEL_ID, "Q4_K_M", config(32768));
  assert.equal(adoptLegacyConfigKey(MODEL_ID, MODEL_ID, "Q4_K_M"), false);
  assert.equal(adoptLegacyConfigKey(MODEL_ID, "", "Q4_K_M"), false);
  assert.equal(
    resolveInitialConfig(MODEL_ID, "Q4_K_M").config.maxSeqLength,
    32768,
  );
  assert.equal(listPerModelConfigs().length, 1);
});

// Saving before deleting briefly holds two copies and silently evicts an unrelated model.
test("moving a legacy key at the entry budget keeps every other model", () => {
  store.clear();
  // The stale record is saved mid-map so it cannot be the eviction victim.
  const half = Math.floor(MAX_ENTRIES / 2);
  for (let i = 0; i < half; i += 1) {
    savePerModelConfig(`org/unrelated-${i}`, "Q4_K_M", config(4096 + i * 128));
  }
  savePerModelConfig(LEGACY_ID, "Q4_K_M", config(32768, "q8_0"));
  for (let i = half; i < MAX_ENTRIES - 1; i += 1) {
    savePerModelConfig(`org/unrelated-${i}`, "Q4_K_M", config(4096 + i * 128));
  }
  assert.equal(listPerModelConfigs().length, MAX_ENTRIES);

  assert.equal(adoptLegacyConfigKey(MODEL_ID, LEGACY_ID, "Q4_K_M"), true);

  const adopted = resolveInitialConfig(MODEL_ID, "Q4_K_M");
  assert.equal(adopted.remembered, true);
  assert.equal(adopted.config.maxSeqLength, 32768);
  assert.equal(adopted.config.kvCacheDtype, "q8_0");
  assert.equal(
    resolveInitialConfig("org/unrelated-0", "Q4_K_M").remembered,
    true,
  );
  assert.equal(resolveInitialConfig(LEGACY_ID, "Q4_K_M").remembered, false);
  assert.equal(listPerModelConfigs().length, MAX_ENTRIES);
});

test("moving a legacy key at the byte budget keeps every other model", () => {
  store.clear();
  for (let i = 0; i < TEMPLATE_MODELS; i += 1) {
    savePerModelConfig(
      `org/template-${i}`,
      "Q4_K_M",
      config(4096, null, BIG_TEMPLATE),
    );
  }
  savePerModelConfig(LEGACY_ID, "Q4_K_M", config(32768, null, BIG_TEMPLATE));
  assert.equal(listPerModelConfigs().length, TEMPLATE_MODELS + 1);

  assert.equal(adoptLegacyConfigKey(MODEL_ID, LEGACY_ID, "Q4_K_M"), true);

  const adopted = resolveInitialConfig(MODEL_ID, "Q4_K_M");
  assert.equal(adopted.remembered, true);
  assert.equal(adopted.config.maxSeqLength, 32768);
  assert.equal(adopted.config.chatTemplateOverride, BIG_TEMPLATE);
  assert.equal(
    resolveInitialConfig("org/template-0", "Q4_K_M").remembered,
    true,
  );
  assert.equal(resolveInitialConfig(LEGACY_ID, "Q4_K_M").remembered, false);
  assert.equal(listPerModelConfigs().length, TEMPLATE_MODELS + 1);
});
