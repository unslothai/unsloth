// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_STT_MODEL,
  RECOMMENDED_STT_MODELS,
  STT_PICKER_MODELS,
  WHISPER_STT_MODELS,
  migrateVoiceSettings,
  normalizeSttModelSource,
  sttModelForHost,
} from "../src/features/settings/stores/stt-model-catalog.ts";

import { readSrc } from "./helpers/kit.ts";

const CUSTOM_ENGINE_BEFORE_TAURI =
  /savedEngine === "custom"\s*\? "custom"\s*:\s*isTauri/;
const EXTERNAL_PROVIDER_FORM_FIELD = /form\.set\("provider_id", providerId\)/;
const EXTERNAL_PRELOAD_GUARD =
  /if \(!usesExternalEndpoint && sessionEngine\) \{[\s\S]*loadSttModel/;
const CONNECTIONS_ENABLED_SEND_GUARD =
  /if \(!providersState\.connectionsEnabled\) \{[\s\S]*form\.set\("provider_id", providerId\)/;
const LEGACY_KEY_FALLBACK =
  /getExternalProviderApiKey\(providerId\)[\s\S]*encryptProviderApiKey\(legacyApiKey\)/;

test("the default dictation model is a recommended one", () => {
  assert.ok(RECOMMENDED_STT_MODELS.has(DEFAULT_STT_MODEL));
});

test("recommended models are listed before the rest", () => {
  const firstOther = STT_PICKER_MODELS.findIndex(
    (model) => !RECOMMENDED_STT_MODELS.has(model),
  );
  const lastRecommended = STT_PICKER_MODELS.reduce(
    (last, model, index) => (RECOMMENDED_STT_MODELS.has(model) ? index : last),
    -1,
  );
  assert.ok(lastRecommended < firstOther);
});

test("a save still on the old default moves to the recommended model", () => {
  const migrated = migrateVoiceSettings({ sttModel: "small" }, 0);
  assert.equal(migrated?.sttModel, DEFAULT_STT_MODEL);
  assert.equal(migrated?.sttModelSource, "user");
});

test("the default is llama.cpp Qwen3-ASR 0.6B, recommended beside its 1.7B", () => {
  assert.equal(DEFAULT_STT_MODEL, "qwen3-asr-0.6b");
  assert.deepEqual(
    [...RECOMMENDED_STT_MODELS],
    ["qwen3-asr-0.6b", "qwen3-asr-1.7b"],
  );
});

test("v1 saves keep their model; only curated Whisper is marked for the host", () => {
  for (const model of [
    "qwen3-asr-0.6b",
    "qwen3-asr-1.7b",
    "audiocpp-moonshine-tiny",
    "org/whisper-x",
  ]) {
    const migrated = migrateVoiceSettings({ sttModel: model, ttsRate: 1.5 }, 1);
    assert.equal(migrated?.sttModel, model, model);
    assert.equal(migrated?.sttModelSource, "user", model);
    assert.equal(migrated?.ttsRate, 1.5);
  }
  for (const model of WHISPER_STT_MODELS) {
    const migrated = migrateVoiceSettings({ sttModel: model }, 1);
    assert.equal(migrated?.sttModel, model);
    assert.equal(migrated?.sttModelSource, "whisper-cpp", model);
  }
  // no saved model is left for merge to fill with the default.
  assert.equal(
    migrateVoiceSettings({ ttsRate: 1 }, 1)?.sttModelSource,
    undefined,
  );
  // v2 saves are not re-migrated.
  assert.equal(
    migrateVoiceSettings({ sttModel: "small", sttModelSource: "user" }, 2)
      ?.sttModelSource,
    "user",
  );
});

test("a saved source is kept when valid and inferred from the model otherwise", () => {
  assert.equal(normalizeSttModelSource("user", DEFAULT_STT_MODEL), "user");
  assert.equal(normalizeSttModelSource("default", "small"), "default");
  assert.equal(normalizeSttModelSource("whisper-cpp", "small"), "whisper-cpp");
  assert.equal(
    normalizeSttModelSource("whisper-cpp", "qwen3-asr-1.7b"),
    "user",
  );
  assert.equal(
    normalizeSttModelSource(undefined, DEFAULT_STT_MODEL),
    "default",
  );
  assert.equal(normalizeSttModelSource("bogus", "qwen3-asr-1.7b"), "user");
});

const host = (transformers: boolean, audiocpp: boolean, mtmd: boolean) => ({
  transformers: { available: transformers },
  audiocpp: { available: audiocpp },
  mtmd: { available: mtmd },
});

test("a retired whisper.cpp pick keeps its key on Transformers, else becomes llama.cpp", () => {
  const pick = { sttModel: "small", sttModelSource: "whisper-cpp" } as const;
  assert.deepEqual(sttModelForHost(pick, host(true, true, true)), {
    sttModel: "small",
    sttModelSource: "user",
  });
  for (const model of WHISPER_STT_MODELS) {
    assert.deepEqual(
      sttModelForHost(
        { sttModel: model, sttModelSource: "whisper-cpp" },
        host(false, true, true),
      ),
      { sttModel: DEFAULT_STT_MODEL, sttModelSource: "default" },
      model,
    );
  }
  assert.deepEqual(sttModelForHost(pick, host(false, true, false)), {
    sttModel: "audiocpp-qwen3-asr-0.6b",
    sttModelSource: "default",
  });
  assert.equal(sttModelForHost(pick, host(false, false, false)), null);
  // an older server without engine blocks leaves it pending.
  assert.equal(sttModelForHost(pick, {}), null);
  // an explicit Whisper pick after the migration is never overridden.
  assert.equal(
    sttModelForHost(
      { sttModel: "small", sttModelSource: "user" },
      host(false, true, true),
    ),
    null,
  );
});

test("the default and any user pick never move with the host", () => {
  for (const sttModelSource of ["default", "user"] as const) {
    for (const engines of [
      host(true, false, true),
      host(true, true, false),
      host(false, false, false),
    ]) {
      assert.equal(
        sttModelForHost(
          { sttModel: DEFAULT_STT_MODEL, sttModelSource },
          engines,
        ),
        null,
      );
    }
  }
});

test("a deliberately chosen model is left alone", () => {
  const migrated = migrateVoiceSettings({ sttModel: "large-v3" }, 0);
  assert.equal(migrated?.sttModel, "large-v3");
});

test("choosing Whisper Small again survives, because v1 does not re-migrate", () => {
  const migrated = migrateVoiceSettings({ sttModel: "small" }, 1);
  assert.equal(migrated?.sttModel, "small");
  assert.equal(migrated?.sttModelSource, "whisper-cpp");
});

test("migration keeps the rest of the save intact", () => {
  const migrated = migrateVoiceSettings(
    { sttModel: "small", dictationLanguage: "ja-JP" },
    0,
  );
  assert.equal(migrated?.dictationLanguage, "ja-JP");
});

test("custom transcription uses a saved connection without loading local STT", () => {
  const storeSource = readSrc(
    "features/settings/stores/voice-settings-store.ts",
  );
  const adapterSource = readSrc(
    "features/chat/adapters/studio-model-dictation-adapter.ts",
  );

  assert.match(storeSource, CUSTOM_ENGINE_BEFORE_TAURI);
  assert.match(adapterSource, EXTERNAL_PROVIDER_FORM_FIELD);
  assert.match(adapterSource, EXTERNAL_PRELOAD_GUARD);
  assert.match(adapterSource, CONNECTIONS_ENABLED_SEND_GUARD);
  assert.match(adapterSource, LEGACY_KEY_FALLBACK);
});

test("the custom dictation connection picker handles deleted and empty connections", () => {
  const voiceTabSource = readSrc("features/settings/tabs/voice-tab.tsx");

  assert.match(
    voiceTabSource,
    /if \(sttProviderId && !hasSelectedSttConnection\) \{\s*setSttProviderId\(""\);/,
  );
  assert.match(
    voiceTabSource,
    /value=\{hasSelectedSttConnection \? sttProviderId : undefined\}/,
  );
  assert.match(
    voiceTabSource,
    /disabled=\{!connectionsEnabled \|\| !hasSttConnections\}/,
  );
  assert.match(voiceTabSource, /"settings\.voice\.dictation\.connectionEmpty"/);
  assert.doesNotMatch(voiceTabSource, /<SelectItem value=\{sttProviderId\}/);
});
