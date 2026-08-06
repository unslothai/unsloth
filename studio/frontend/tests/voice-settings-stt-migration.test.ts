// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_STT_MODEL,
  RECOMMENDED_STT_MODELS,
  STT_MODELS,
  migrateVoiceSettings,
} from "../src/features/settings/stores/stt-model-catalog.ts";

test("the default dictation model is a recommended one", () => {
  assert.ok(RECOMMENDED_STT_MODELS.has(DEFAULT_STT_MODEL));
});

test("recommended models are listed before the rest", () => {
  const firstOther = STT_MODELS.findIndex(
    (model) => !RECOMMENDED_STT_MODELS.has(model),
  );
  const lastRecommended = STT_MODELS.reduce(
    (last, model, index) => (RECOMMENDED_STT_MODELS.has(model) ? index : last),
    -1,
  );
  assert.ok(lastRecommended < firstOther);
});

test("a save still on the old default moves to the recommended model", () => {
  const migrated = migrateVoiceSettings({ sttModel: "small" }, 0);
  assert.equal(migrated?.sttModel, DEFAULT_STT_MODEL);
});

test("a deliberately chosen model is left alone", () => {
  const migrated = migrateVoiceSettings({ sttModel: "large-v3" }, 0);
  assert.equal(migrated?.sttModel, "large-v3");
});

test("choosing Whisper Small again survives, because v1 does not re-migrate", () => {
  const migrated = migrateVoiceSettings({ sttModel: "small" }, 1);
  assert.equal(migrated?.sttModel, "small");
});

test("migration keeps the rest of the save intact", () => {
  const migrated = migrateVoiceSettings(
    { sttModel: "small", dictationLanguage: "ja-JP" },
    0,
  );
  assert.equal(migrated?.dictationLanguage, "ja-JP");
});
