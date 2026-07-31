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
    tensorParallel: false,
    chatTemplateOverride,
  };
}

// MAX_ENTRIES in per-model-config.ts, which does not export it.
const MAX_ENTRIES = 500;
// MAX_PER_MODEL_CONFIG_STORAGE_BYTES is 1 MiB, so a handful of models carrying a large
// chat template override sits against the byte budget well before the entry budget.
const BIG_TEMPLATE = "x".repeat(60_000);
const TEMPLATE_MODELS = 16;

// The values have to be asserted, not just the key. Passing the config where the quant goes
// writes an all-defaults record and reports success, after which the legacy record is dropped
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

// A save before the delete holds two copies at once, one entry over a full map, and
// savePerModelConfig then evicts the oldest unrelated model silently. This path passes no
// eviction list, so that model's server override outlives anything the UI could forget.
test("moving a legacy key at the entry budget keeps every other model", () => {
  store.clear();
  // A full map, with the stale record saved partway through so it is not the oldest entry
  // and so cannot be the one eviction happens to take.
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
  // The oldest entry is the first eviction would take, and every model is still there: the
  // move traded one key for another rather than adding a second copy.
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
