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
  resolveInitialConfig,
  resolveResidentInitialConfig,
  savePerModelConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

// What /api/inference/status reports as model_identifier after an API auto-switch: the concrete
// snapshot path the resolver index handed the loader. The repo id is what the Hub keys this
// model's settings by (modelConfigIdentity) and what status.active_model reports.
const SNAPSHOT_PATH =
  "/home/u/.cache/huggingface/hub/models--unsloth--Repo-GGUF/snapshots/2f1c9ab";
const REPO_ID = "unsloth/Repo-GGUF";

function config(nParallel: number | null, maxSeqLength: number | null = null) {
  return {
    customContextLength: null,
    maxSeqLength,
    kvCacheDtype: null,
    speculativeType: null,
    specDraftNMax: null,
    nParallel,
    tensorParallel: false,
    chatTemplateOverride: null,
  };
}

test("the resident model's repo-keyed slots are found through its snapshot path", () => {
  store.clear();
  savePerModelConfig(REPO_ID, "Q4_K_M", config(4));

  // The plain lookup is what status adoption used to do, and it misses.
  assert.equal(resolveInitialConfig(SNAPSHOT_PATH, "Q4_K_M").remembered, false);

  const resolved = resolveResidentInitialConfig(SNAPSHOT_PATH, "Q4_K_M");
  assert.equal(resolved.remembered, true);
  // Without this the control blanks on the model change and the next save writes the
  // blank back over the saved record (locally and through the server mirror).
  assert.equal(resolved.config.nParallel, 4);
});

test("a record under the raw identifier still wins over the repo alias", () => {
  store.clear();
  savePerModelConfig(SNAPSHOT_PATH, "Q4_K_M", config(2));
  savePerModelConfig(REPO_ID, "Q4_K_M", config(8));

  assert.equal(
    resolveResidentInitialConfig(SNAPSHOT_PATH, "Q4_K_M").config.nParallel,
    2,
  );
});

test("the quant still separates two variants of one cached repo", () => {
  store.clear();
  savePerModelConfig(REPO_ID, "Q4_K_M", config(4));

  assert.equal(
    resolveResidentInitialConfig(SNAPSHOT_PATH, "Q8_0").remembered,
    false,
  );
});

test("a stem two models can share is never read as an alias", () => {
  store.clear();
  // A standalone GGUF is keyed by its own path; "Repo-Q4_K_M" is the stem both of these
  // files collapse onto, so adopting it would apply one file's settings to the other.
  savePerModelConfig("Repo-Q4_K_M", null, config(4));

  assert.equal(
    resolveResidentInitialConfig("/srv/models/a/Repo-Q4_K_M.gguf", null)
      .remembered,
    false,
  );
  assert.equal(
    resolveResidentInitialConfig("/srv/models/b/Repo-Q4_K_M.gguf", null)
      .remembered,
    false,
  );
});

test("a repo id resolves exactly as before", () => {
  store.clear();
  savePerModelConfig(REPO_ID, "Q4_K_M", config(4));

  assert.equal(
    resolveResidentInitialConfig(REPO_ID, "Q4_K_M").config.nParallel,
    4,
  );
  assert.equal(
    resolveResidentInitialConfig("unsloth/Other-GGUF", "Q4_K_M").remembered,
    false,
  );
});
