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

// A snapshot path is what an older release keyed a repo cached outside the active HF cache
// by; the repo id is what it is keyed by now.
const LEGACY_ID = "/home/u/.cache/models/snapshots/2f1c9ab";
const MODEL_ID = "unsloth/Repo-GGUF";

function config(maxSeqLength: number, kvCacheDtype: string | null = null) {
  return {
    customContextLength: null,
    maxSeqLength,
    kvCacheDtype,
    speculativeType: null,
    specDraftNMax: null,
    nParallel: null,
    tensorParallel: false,
    chatTemplateOverride: null,
  };
}

// The values have to be asserted, not just the key. Passing the config where the quant
// goes writes an all-defaults record (normalize rejects the string, isDefaultConfig then
// takes the delete branch) and reports success, after which the legacy record is dropped
// anyway, so a key-count or source-substring check still reads as a successful move.
test("a legacy-keyed config moves to the current id with its values", () => {
  store.clear();
  savePerModelConfig(LEGACY_ID, "Q4_K_M", config(32768, "q8_0"));

  assert.equal(adoptLegacyConfigKey(MODEL_ID, LEGACY_ID, "Q4_K_M"), true);

  const adopted = resolveInitialConfig(MODEL_ID, "Q4_K_M");
  assert.equal(adopted.remembered, true);
  assert.equal(adopted.config.maxSeqLength, 32768);
  assert.equal(adopted.config.kvCacheDtype, "q8_0");
  // The stale record goes, so exactly one survives and nothing reads the old key.
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
  // No legacy record at all.
  assert.equal(adoptLegacyConfigKey(MODEL_ID, LEGACY_ID, "Q4_K_M"), false);

  savePerModelConfig(MODEL_ID, "Q4_K_M", config(32768));
  // The two ids are the same, or there is no older id to move from, so the record the
  // caller is about to read must be left exactly where it is.
  assert.equal(adoptLegacyConfigKey(MODEL_ID, MODEL_ID, "Q4_K_M"), false);
  assert.equal(adoptLegacyConfigKey(MODEL_ID, "", "Q4_K_M"), false);
  assert.equal(
    resolveInitialConfig(MODEL_ID, "Q4_K_M").config.maxSeqLength,
    32768,
  );
  assert.equal(listPerModelConfigs().length, 1);
});
