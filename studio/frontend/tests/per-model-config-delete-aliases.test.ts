// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A forget has to reach both spellings of a cached repo, because the server's remove does.
// Leaving the other one behind hides the row until the model's next save mirrors it back.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const {
  deletePerModelConfigAliases,
  listPerModelConfigs,
  resolveInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { ggufQuantLabel } = await import(
  "../src/features/model-picker/model-config/model-identity.ts"
);

const REPO_ID = "unsloth/Repo-GGUF";
// What the loader opens and an older release keyed the row by.
const SNAPSHOT_PATH =
  "/home/u/.cache/huggingface/hub/models--unsloth--Repo-GGUF/snapshots/2f1c9ab";
const OTHER_PATH = "/home/u/models/Repo-Q4_K_M.gguf";
// A loose file is keyed by its bare path now; the picker once keyed it by this label.
const LOOSE_PATH = "/home/u/models/Qwen3-4B-Q4_K_M.gguf";
const QUANT = "Q4_K_M";

function config(maxSeqLength: number) {
  return {
    customContextLength: null,
    maxSeqLength,
    kvCacheDtype: null,
    speculativeType: null,
    specDraftNMax: null,
    nParallel: null,
    nBatch: null,
    nUbatch: null,
    tensorParallel: false,
    disableVision: false,
    chatTemplateOverride: null,
  };
}

test("forgetting the snapshot-path row also drops the repo-id record", () => {
  store.clear();
  // adoptLegacyConfigKey moved this browser's record here; the server row is still the path.
  savePerModelConfig(REPO_ID, QUANT, config(32768));

  assert.equal(deletePerModelConfigAliases(SNAPSHOT_PATH, QUANT), true);

  assert.equal(resolveInitialConfig(REPO_ID, QUANT).remembered, false);
  assert.equal(listPerModelConfigs().length, 0);
});

test("forgetting the repo-id row also drops a record still keyed by the path", () => {
  store.clear();
  savePerModelConfig(SNAPSHOT_PATH, QUANT, config(32768));

  assert.equal(deletePerModelConfigAliases(REPO_ID, QUANT), true);

  assert.equal(resolveInitialConfig(SNAPSHOT_PATH, QUANT).remembered, false);
  assert.equal(listPerModelConfigs().length, 0);
});

test("another quant of the same repo is left alone", () => {
  store.clear();
  savePerModelConfig(REPO_ID, "Q8_0", config(4096));

  assert.equal(deletePerModelConfigAliases(SNAPSHOT_PATH, QUANT), true);

  assert.equal(resolveInitialConfig(REPO_ID, "Q8_0").remembered, true);
});

test("an ordinary local path is keyed by its path and takes nothing else with it", () => {
  store.clear();
  savePerModelConfig(REPO_ID, QUANT, config(32768));
  savePerModelConfig(OTHER_PATH, QUANT, config(4096));

  assert.equal(deletePerModelConfigAliases(OTHER_PATH, QUANT), true);

  assert.equal(resolveInitialConfig(OTHER_PATH, QUANT).remembered, false);
  assert.equal(resolveInitialConfig(REPO_ID, QUANT).remembered, true);
});

test("a bare entry with no quant has no other spelling to reach", () => {
  store.clear();
  savePerModelConfig(REPO_ID, null, config(32768));
  savePerModelConfig(SNAPSHOT_PATH, null, config(4096));

  assert.equal(deletePerModelConfigAliases(SNAPSHOT_PATH, null), true);

  assert.equal(resolveInitialConfig(REPO_ID, null).remembered, true);
});

test("a bare-path forget also drops the label a loose .gguf used to be keyed by", () => {
  store.clear();
  const label = ggufQuantLabel("Qwen3-4B-Q4_K_M.gguf");
  savePerModelConfig(LOOSE_PATH, label, config(32768));

  assert.equal(deletePerModelConfigAliases(LOOSE_PATH, null), true);

  // resolveChatModelSwitchTarget promotes exactly this record, and the next save
  // would mirror it back to the server the row was just removed from.
  assert.equal(resolveInitialConfig(LOOSE_PATH, label).remembered, false);
  assert.equal(listPerModelConfigs().length, 0);
});

test("forgetting one quant of a loose .gguf leaves the bare-path record alone", () => {
  store.clear();
  savePerModelConfig(LOOSE_PATH, null, config(4096));

  assert.equal(deletePerModelConfigAliases(LOOSE_PATH, "Q4_K_M"), true);

  assert.equal(resolveInitialConfig(LOOSE_PATH, null).remembered, true);
});

test("a repo id is not read as a loose file", () => {
  store.clear();
  savePerModelConfig(REPO_ID, "Q4_K_M", config(32768));

  assert.equal(deletePerModelConfigAliases(REPO_ID, null), true);

  assert.equal(resolveInitialConfig(REPO_ID, "Q4_K_M").remembered, true);
});
