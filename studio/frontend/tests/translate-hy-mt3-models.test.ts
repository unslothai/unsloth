// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEDICATED_TRANSLATE_MODELS,
  DEFAULT_TRANSLATE_MODEL,
  HY_MT_SAMPLING,
  dedicatedTranslateModel,
  migrateTranslateModelId,
  offeredTranslateModelId,
  offeredTranslateModels,
  translationSampling,
} from "../src/features/translate/models.ts";
import { isTranslationOnlyRepo } from "../src/lib/translation-models.ts";

const ids = (models: readonly { id: string }[]) => models.map((m) => m.id);

test("the default is Hy-MT3 1.8B as a GGUF from unslothai", () => {
  const model = dedicatedTranslateModel(DEFAULT_TRANSLATE_MODEL);
  assert.equal(model?.repo, "unslothai/Hy-MT3-1.8B-GGUF");
  assert.equal(model?.variant, "Q4_K_M");
});

test("FP8 uploads are offered only where the GPU runs FP8", () => {
  assert.deepEqual(ids(offeredTranslateModels([])), [
    "hy-mt3-1.8b",
    "hy-mt3-7b",
    "hy-mt3-30b-a3b",
  ]);
  assert.deepEqual(ids(offeredTranslateModels(["int8"])), [
    "hy-mt3-1.8b",
    "hy-mt3-7b",
    "hy-mt3-30b-a3b",
  ]);
  assert.deepEqual(
    ids(offeredTranslateModels(["fp8", "nvfp4"])),
    ids(DEDICATED_TRANSLATE_MODELS),
  );
});

test("every FP8 upload is a whole safetensors repo with a GGUF sibling", () => {
  for (const model of DEDICATED_TRANSLATE_MODELS.filter((m) => m.fp8)) {
    assert.equal(model.variant, null);
    assert.match(model.repo, /^unslothai\/Hy-MT3-.+-FP8$/);
    const sibling = dedicatedTranslateModel(
      offeredTranslateModelId(model.id, []),
    );
    assert.equal(sibling?.fp8, undefined);
    assert.equal(sibling?.repo, model.repo.replace(/-FP8$/, "-GGUF"));
    assert.deepEqual(sibling?.sampling, model.sampling);
  }
});

test("a saved FP8 pick falls back to GGUF only on a device without FP8", () => {
  assert.equal(offeredTranslateModelId("hy-mt3-7b-fp8", []), "hy-mt3-7b");
  assert.equal(
    offeredTranslateModelId("hy-mt3-7b-fp8", ["fp8"]),
    "hy-mt3-7b-fp8",
  );
  assert.equal(offeredTranslateModelId("hy-mt3-7b", []), "hy-mt3-7b");
  assert.equal(offeredTranslateModelId("chat", []), "chat");
});

test("a Hy-MT2 selection saved earlier moves to the same Hy-MT3 size", () => {
  assert.equal(migrateTranslateModelId("hy-mt2-1.8b"), "hy-mt3-1.8b");
  assert.equal(migrateTranslateModelId("hy-mt2-7b"), "hy-mt3-7b");
  assert.equal(migrateTranslateModelId("custom"), "custom");
  assert.equal(migrateTranslateModelId(undefined), undefined);
});

test("30B-A3B uses its own model-card sampling, with top_k sent as disabled", () => {
  const moe = dedicatedTranslateModel("hy-mt3-30b-a3b")?.sampling;
  assert.deepEqual(moe, {
    temperature: 0.7,
    top_p: 1,
    top_k: -1,
    repetition_penalty: 1,
  });
  assert.deepEqual(translationSampling("UNSLOTHAI/Hy-MT3-30B-A3B-FP8"), moe);
  assert.deepEqual(
    translationSampling("unslothai/Hy-MT3-7B-GGUF"),
    HY_MT_SAMPLING,
  );
  assert.deepEqual(
    translationSampling("unsloth/Hy-MT2-7B-GGUF"),
    HY_MT_SAMPLING,
  );
});

test("Hy-MT2 copies still on disk stay translation models", () => {
  assert.equal(isTranslationOnlyRepo("unsloth/Hy-MT2-7B-GGUF"), true);
  assert.equal(isTranslationOnlyRepo("unslothai/hy-mt3-30b-a3b-fp8"), true);
  assert.equal(isTranslationOnlyRepo("unsloth/Qwen3-8B-GGUF"), false);
});
