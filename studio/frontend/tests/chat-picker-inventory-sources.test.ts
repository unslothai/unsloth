// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  completeCachedModelKeys,
  isHfCacheDuplicate,
  pickerLocalModelMatchesQuery,
} from "../src/features/model-picker/inventory/chat-picker-inventory-sources.ts";

const SOURCE_HF_CACHE = "hf_cache";
const SOURCE_MODELS_DIR = "models_dir";
const MODEL_FORMAT = "gguf";
const SAFETENSORS_FORMAT = "safetensors";
const ACTIVE_REPO_ID = "owao/Nanbeige4.2-3B-GGUF";
const UNSEEN_REPO_ID = "other/never-cached-GGUF";

function cachedRow(
  repoId: string,
  state: { partial?: boolean; liveDownload?: boolean } = {},
  modelFormat = MODEL_FORMAT,
) {
  return { repoId, modelFormat, ...state };
}

function localRow(
  repoId: string | null,
  source = SOURCE_HF_CACHE,
  modelFormat = MODEL_FORMAT,
) {
  return { source, repoId, modelFormat };
}

test("keys only fully materialized cached models", () => {
  const keys = completeCachedModelKeys([
    cachedRow(ACTIVE_REPO_ID),
    cachedRow("partial/model-GGUF", { partial: true }),
    cachedRow("live/model-GGUF", { liveDownload: true }),
  ]);

  assert.equal(keys.size, 1);
  assert.equal(isHfCacheDuplicate(localRow(ACTIVE_REPO_ID), keys), true);
});

test("dedupes an hf_cache row only against a matching complete cached twin", () => {
  // Keying is whitespace- and case-insensitive on the repo id.
  const keys = completeCachedModelKeys([cachedRow(` ${ACTIVE_REPO_ID} `)]);

  assert.equal(isHfCacheDuplicate(localRow(ACTIVE_REPO_ID), keys), true);
  // A different local source is never a cache duplicate, even at the same repo.
  assert.equal(
    isHfCacheDuplicate(localRow(ACTIVE_REPO_ID, SOURCE_MODELS_DIR), keys),
    false,
  );
  // Format is part of identity: a safetensors local copy is not the cached GGUF.
  assert.equal(
    isHfCacheDuplicate(
      localRow(ACTIVE_REPO_ID, SOURCE_HF_CACHE, SAFETENSORS_FORMAT),
      keys,
    ),
    false,
  );
  // A repo-less local row cannot be matched against the cache.
  assert.equal(isHfCacheDuplicate(localRow(null), keys), false);
  // An hf_cache row with no cached twin stays visible.
  assert.equal(isHfCacheDuplicate(localRow(UNSEEN_REPO_ID), keys), false);
});

test("an incomplete cached entry does not hide its hf_cache copy", () => {
  const keys = completeCachedModelKeys([
    cachedRow(ACTIVE_REPO_ID, { partial: true }),
  ]);

  assert.equal(isHfCacheDuplicate(localRow(ACTIVE_REPO_ID), keys), false);
});

test("normalizes local search and accepts an empty query", () => {
  const model = {
    id: ACTIVE_REPO_ID,
    display_name: ACTIVE_REPO_ID,
    model_id: ACTIVE_REPO_ID,
  };

  assert.equal(pickerLocalModelMatchesQuery(model, ""), true);
  assert.equal(pickerLocalModelMatchesQuery(model, " NANBEIGE4_2-3B "), true);
  assert.equal(pickerLocalModelMatchesQuery(model, "missing"), false);
});

test("falls back to display name when repository metadata is absent", () => {
  const model = {
    id: "/models/local-model",
    display_name: "Local Model",
    model_id: null,
  };

  assert.equal(pickerLocalModelMatchesQuery(model, "localmodel"), true);
  assert.equal(pickerLocalModelMatchesQuery(model, "absent"), false);
});
