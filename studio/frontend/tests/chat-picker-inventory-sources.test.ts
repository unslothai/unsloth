// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildPickerLocalModels,
  pickerLocalModelMatchesQuery,
  type PickerCachedRow,
  type PickerLocalRow,
} from "../src/features/model-picker/inventory/chat-picker-inventory-sources.ts";

const ACTIVE_REPO_ID = "owao/Nanbeige4.2-3B-GGUF";
const DUPLICATE_REPO_ID = "other/already-listed-GGUF";
const INACTIVE_REPO_ID = "legacy/inactive-GGUF";
const INACTIVE_LOAD_PATH = "/legacy/snapshots/inactive";
const MODEL_FORMAT = "gguf";
const SAFETENSORS_FORMAT = "safetensors";
const SOURCE_CUSTOM = "custom";
const SOURCE_HF_CACHE = "hf_cache";
const SOURCE_MODELS_DIR = "models_dir";
const SOURCE_OLLAMA = "ollama";
const REPO_SEPARATOR = "/";

const CHAT_CAPABILITIES = {
  canChat: true,
};

function cachedRow(
  repoId: string,
  state: Pick<PickerCachedRow, "partial" | "liveDownload"> = {},
  modelFormat = MODEL_FORMAT,
): PickerCachedRow {
  return {
    repoId,
    modelFormat,
    ...state,
  };
}

function localRow(
  repoId: string,
  loadId: string,
  source: PickerLocalRow["source"] = SOURCE_HF_CACHE,
  canChat = true,
): PickerLocalRow {
  return {
    loadId,
    repoId,
    title: repoId.split(REPO_SEPARATOR)[1],
    source,
    modelId: repoId,
    displayName: repoId,
    path: loadId,
    modelFormat: MODEL_FORMAT,
    capabilities: canChat ? CHAT_CAPABILITIES : { canChat: false },
    updatedAt: null,
  };
}

test("projects searchable Hugging Face cache fallbacks without duplicates", () => {
  const models = buildPickerLocalModels(
    [cachedRow(` ${DUPLICATE_REPO_ID} `)],
    [
      localRow(ACTIVE_REPO_ID, ACTIVE_REPO_ID),
      localRow(INACTIVE_REPO_ID, INACTIVE_LOAD_PATH),
      localRow(DUPLICATE_REPO_ID, DUPLICATE_REPO_ID),
    ],
  );

  assert.deepEqual(
    models.map((model) => ({
      id: model.id,
      modelId: model.model_id,
      source: model.source,
    })),
    [
      {
        id: ACTIVE_REPO_ID,
        modelId: ACTIVE_REPO_ID,
        source: SOURCE_MODELS_DIR,
      },
      {
        id: INACTIVE_LOAD_PATH,
        modelId: INACTIVE_REPO_ID,
        source: SOURCE_MODELS_DIR,
      },
    ],
  );
  assert.equal(
    models.filter((model) => pickerLocalModelMatchesQuery(model, "nanbe"))
      .length,
    1,
  );
  assert.equal(
    models.filter((model) => pickerLocalModelMatchesQuery(model, "inactive"))
      .length,
    1,
  );
});

test("keeps usable local sources and incomplete cache fallbacks", () => {
  const partialRepoId = "partial/model-GGUF";
  const liveRepoId = "live/model-GGUF";
  const customRepoId = "custom/model";
  const nonChatRepoId = "weights/without-chat";

  const models = buildPickerLocalModels(
    [
      cachedRow(partialRepoId, { partial: true }),
      cachedRow(liveRepoId, { liveDownload: true }),
    ],
    [
      localRow(partialRepoId, partialRepoId),
      localRow(liveRepoId, liveRepoId),
      localRow(customRepoId, customRepoId, SOURCE_CUSTOM),
      localRow(nonChatRepoId, nonChatRepoId, SOURCE_MODELS_DIR, false),
      localRow("ollama/unsupported", "ollama/unsupported", SOURCE_OLLAMA),
    ],
  );

  assert.deepEqual(
    models.map((model) => model.model_id),
    [partialRepoId, liveRepoId, customRepoId],
  );
  assert.equal(models.at(-1)?.source, SOURCE_CUSTOM);
});

test("keeps a local format when another format is already cached", () => {
  const multiFormatRepoId = "formats/model";
  const models = buildPickerLocalModels(
    [cachedRow(multiFormatRepoId, {}, SAFETENSORS_FORMAT)],
    [localRow(multiFormatRepoId, multiFormatRepoId)],
  );

  assert.equal(models.length, 1);
  assert.equal(models[0]?.model_format, MODEL_FORMAT);
});

test("normalizes local search and accepts an empty query", () => {
  const [model] = buildPickerLocalModels(
    [],
    [localRow(ACTIVE_REPO_ID, ACTIVE_REPO_ID)],
  );

  assert.ok(model);
  assert.equal(pickerLocalModelMatchesQuery(model, ""), true);
  assert.equal(pickerLocalModelMatchesQuery(model, " NANBEIGE4_2-3B "), true);
  assert.equal(pickerLocalModelMatchesQuery(model, "missing"), false);
});

test("falls back to local titles when repository metadata is absent", () => {
  const localTitle = "Local Model";
  const localPath = "/models/local-model";
  const [model] = buildPickerLocalModels(
    [],
    [
      {
        ...localRow("placeholder/repo", localPath, SOURCE_MODELS_DIR),
        repoId: null,
        modelId: null,
        displayName: undefined,
        title: localTitle,
      },
    ],
  );

  assert.ok(model);
  assert.equal(model.display_name, localTitle);
  assert.equal(model.model_id, null);
  assert.equal(pickerLocalModelMatchesQuery(model, "localmodel"), true);
});
