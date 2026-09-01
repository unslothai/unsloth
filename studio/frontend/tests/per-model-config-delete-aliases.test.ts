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
  findModelOverrideKeyOwner,
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

test("the same forget from the legacy side drops the bare-path record too", () => {
  store.clear();
  savePerModelConfig(LOOSE_PATH, null, config(4096));

  // Both spellings name one file, and _bare_model_id splits a path key the same way, so
  // the server's remove takes the bare path here as well.
  assert.equal(deletePerModelConfigAliases(LOOSE_PATH, "Q4_K_M"), true);

  assert.equal(resolveInitialConfig(LOOSE_PATH, null).remembered, false);
});

test("a repo id is not read as a loose file", () => {
  store.clear();
  savePerModelConfig(REPO_ID, "Q4_K_M", config(32768));

  assert.equal(deletePerModelConfigAliases(REPO_ID, null), true);

  assert.equal(resolveInitialConfig(REPO_ID, "Q4_K_M").remembered, true);
});

test("forgetting a repo's last quant also drops the bare record a load falls back to", () => {
  store.clear();
  savePerModelConfig(REPO_ID, null, config(4096));
  savePerModelConfig(REPO_ID, QUANT, config(32768));

  assert.equal(deletePerModelConfigAliases(REPO_ID, QUANT), true);

  assert.equal(resolveInitialConfig(REPO_ID, null).remembered, false);
  assert.equal(listPerModelConfigs().length, 0);
});

test("forgetting one quant of several leaves the bare record for the rest", () => {
  store.clear();
  savePerModelConfig(REPO_ID, null, config(4096));
  savePerModelConfig(REPO_ID, QUANT, config(32768));
  savePerModelConfig(REPO_ID, "Q8_0", config(16384));

  assert.equal(deletePerModelConfigAliases(REPO_ID, QUANT), true);

  // The bare record is what Q8_0 inherits; taking it would forget a quant nobody asked about.
  assert.equal(resolveInitialConfig(REPO_ID, null).remembered, true);
  assert.equal(resolveInitialConfig(REPO_ID, "Q8_0").remembered, true);
});

test("forgetting a bare row does not reach into the model's quants", () => {
  store.clear();
  savePerModelConfig(REPO_ID, null, config(4096));
  savePerModelConfig(REPO_ID, QUANT, config(32768));

  assert.equal(deletePerModelConfigAliases(REPO_ID, null), true);

  assert.equal(resolveInitialConfig(REPO_ID, QUANT).remembered, true);
});

// A key joins the id and the variant with a colon, so a path that meets a directory-qualified
// variant cannot be split back apart: a colon is legal in a POSIX filename. The record knows.
test("a directory-qualified variant under a path resolves to its stored record", () => {
  store.clear();
  savePerModelConfig(
    "/home/u/models/repo",
    "distilled/model-Q6_K",
    config(32768),
  );

  assert.deepEqual(
    findModelOverrideKeyOwner("/home/u/models/repo:distilled/model-Q6_K"),
    { modelId: "/home/u/models/repo", ggufVariant: "distilled/model-q6_k" },
  );
});

test("the server row's casing still finds the lowercased record", () => {
  store.clear();
  savePerModelConfig(REPO_ID, QUANT, config(32768));

  assert.deepEqual(findModelOverrideKeyOwner(`${REPO_ID}:q4_k_m`), {
    modelId: REPO_ID.toLowerCase(),
    ggufVariant: "q4_k_m",
  });
});

test("a key this browser has no record for resolves to nothing", () => {
  store.clear();
  savePerModelConfig(REPO_ID, QUANT, config(32768));

  assert.equal(findModelOverrideKeyOwner("unsloth/Other-GGUF:Q8_0"), null);
});

test("a bare key resolves to the record with no variant", () => {
  store.clear();
  savePerModelConfig(REPO_ID, null, config(4096));

  assert.deepEqual(findModelOverrideKeyOwner(REPO_ID), {
    modelId: REPO_ID.toLowerCase(),
    ggufVariant: null,
  });
});
