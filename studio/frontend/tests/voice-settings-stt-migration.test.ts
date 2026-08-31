// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DEFAULT_STT_MODEL,
  RECOMMENDED_STT_MODELS,
  STT_MODELS,
  migrateVoiceSettings,
} from "../src/features/settings/stores/stt-model-catalog.ts";

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

test("custom transcription uses a saved connection without loading local STT", () => {
  const storeSource = readFileSync(
    new URL(
      "../src/features/settings/stores/voice-settings-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const adapterSource = readFileSync(
    new URL(
      "../src/features/chat/adapters/studio-model-dictation-adapter.ts",
      import.meta.url,
    ),
    "utf8",
  );

  assert.match(storeSource, CUSTOM_ENGINE_BEFORE_TAURI);
  assert.match(adapterSource, EXTERNAL_PROVIDER_FORM_FIELD);
  assert.match(adapterSource, EXTERNAL_PRELOAD_GUARD);
  assert.match(adapterSource, CONNECTIONS_ENABLED_SEND_GUARD);
  assert.match(adapterSource, LEGACY_KEY_FALLBACK);
});

test("the custom dictation connection picker handles deleted and empty connections", () => {
  const voiceTabSource = readFileSync(
    new URL(
      "../src/features/settings/tabs/voice-tab.tsx",
      import.meta.url,
    ),
    "utf8",
  );

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
  assert.match(
    voiceTabSource,
    /"settings\.voice\.dictation\.connectionEmpty"/,
  );
  assert.doesNotMatch(
    voiceTabSource,
    /<SelectItem value=\{sttProviderId\}/,
  );
});
