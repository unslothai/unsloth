// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
    reasoningBudget: -1,
    reasoningBudgetMessage: "",
    nBatch: null,
    nUbatch: null,
    tensorParallel: false,
    disableVision: false,
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

for (const modelId of [
  "/home/u/.lmstudio/models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
  String.raw`C:\Users\u\.lmstudio\models\unsloth\Qwen3-0.6B-GGUF\Qwen3-0.6B-Q4_K_M.gguf`,
  "Qwen3-0.6B-Q4_K_M.gguf",
]) {
  test(`a resident standalone GGUF restores the settings saved by the picker: ${modelId}`, () => {
    store.clear();
    savePerModelConfig(modelId, null, {
      ...config(2),
      customContextLength: 4096,
    });

    const resolved = resolveResidentInitialConfig(modelId, "Q4_K_M");
    assert.equal(resolved.remembered, true);
    assert.equal(resolved.config.customContextLength, 4096);
    assert.equal(resolved.config.nParallel, 2);
  });
}

for (const modelId of [
  "/home/u/.lmstudio/models/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
  "Qwen3-0.6B-Q4_K_M.gguf",
]) {
  test(`a record the picker keyed by the quant label before #7473 is still read: ${modelId}`, () => {
    store.clear();
    savePerModelConfig(modelId, "Q4_K_M", config(5));

    // Only the label is on disk, so the fallback is what answers.
    const resolved = resolveResidentInitialConfig(modelId, "Q4_K_M");
    assert.equal(resolved.remembered, true);
    assert.equal(resolved.config.nParallel, 5);

    // A record under the path still wins over the older labelled one.
    savePerModelConfig(modelId, null, config(6));
    assert.equal(
      resolveResidentInitialConfig(modelId, "Q4_K_M").config.nParallel,
      6,
    );
  });
}

for (const modelId of [
  "/home/u/.lmstudio/models/unsloth/Qwen3-0.6B-GGUF",
  "org/model.gguf",
]) {
  test(`a model containing several GGUF variants keeps their settings separate: ${modelId}`, () => {
    store.clear();
    savePerModelConfig(modelId, null, config(8));
    savePerModelConfig(modelId, "Q4_K_M", config(2));
    savePerModelConfig(modelId, "Q8_0", config(4));

    assert.equal(resolveResidentInitialConfig(modelId, "Q4_K_M").config.nParallel, 2);
    assert.equal(resolveResidentInitialConfig(modelId, "Q8_0").config.nParallel, 4);
    assert.equal(resolveResidentInitialConfig(modelId, "Q6_K").remembered, false);
  });
}

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

const composer = readFileSync(
  new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
  "utf8",
);
const ensureModelLoaded = composer.slice(
  composer.indexOf("async function ensureModelLoaded("),
  composer.indexOf("if (handle1 && model1?.id)"),
);

test("compare ensureModelLoaded recovers a repo-keyed pin through a snapshot path", () => {
  store.clear();
  savePerModelConfig(REPO_ID, "Q4_K_M", {
    ...config(null),
    customContextLength: 8192,
  });
  assert.equal(resolveInitialConfig(SNAPSHOT_PATH, "Q4_K_M").remembered, false);
  assert.equal(
    resolveResidentInitialConfig(SNAPSHOT_PATH, "Q4_K_M").config
      .customContextLength,
    8192,
  );
  assert.match(
    ensureModelLoaded,
    /resolveResidentInitialConfig\(\s*sel\.id,\s*sel\.ggufVariant \?\? null,?\s*\)/,
  );
  assert.doesNotMatch(
    ensureModelLoaded,
    /resolveInitialConfig\(\s*sel\.id,/,
  );
});
