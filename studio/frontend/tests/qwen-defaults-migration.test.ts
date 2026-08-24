// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { PersistedChatSettings } from "../src/features/chat/api/chat-settings-api.ts";
import { migrateLegacyQwenDefaults } from "../src/features/chat/utils/qwen-defaults-migration.ts";

const QWEN38 = "unsloth/Qwen3.8-27B-GGUF";
const LEGACY_SNAPSHOT = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.01,
  repetitionPenalty: 1.0,
  presencePenalty: 0.0,
  maxTokens: 8192,
  systemPrompt: "",
  systemVariables: "",
  fastMode: false,
};

function settingsFor(
  modelId: string,
  entry: PersistedChatSettings["inferenceParams"] = LEGACY_SNAPSHOT,
): PersistedChatSettings {
  return {
    activePreset: "Default",
    activePresetSource: "builtin-default",
    inferenceParams: {
      temperature: 0.6,
      topP: 0.95,
      minP: 0.01,
      presencePenalty: 0.0,
      maxTokens: 8192,
    },
    inferenceParamsByModel: { [modelId]: entry },
  };
}

test("migrates the complete legacy Qwen3.8 default snapshot", () => {
  const migrated = migrateLegacyQwenDefaults(
    settingsFor(QWEN38),
    QWEN38,
    true,
    true,
  );

  assert.deepEqual(migrated.migratedModelIds, [QWEN38]);
  assert.equal(
    migrated.settings.inferenceParamsByModel?.[QWEN38]?.presencePenalty,
    1.5,
  );
  assert.equal(migrated.settings.inferenceParamsByModel?.[QWEN38]?.minP, 0);
  assert.equal(migrated.settings.inferenceParams?.presencePenalty, 0);
  assert.equal(migrated.settings.inferenceParams?.minP, 0.01);
  assert.deepEqual(migrated.patch, {
    inferenceParamsByModel: {
      [QWEN38]: { minP: 0, presencePenalty: 1.5 },
    },
  });
});

test("migrates temperature and top-p for a non-thinking session", () => {
  const migrated = migrateLegacyQwenDefaults(
    settingsFor(QWEN38),
    QWEN38,
    false,
    true,
  );

  assert.equal(
    migrated.settings.inferenceParamsByModel?.[QWEN38]?.temperature,
    0.7,
  );
  assert.equal(migrated.settings.inferenceParamsByModel?.[QWEN38]?.topP, 0.8);
  assert.deepEqual(migrated.patch, {
    inferenceParamsByModel: {
      [QWEN38]: {
        temperature: 0.7,
        topP: 0.8,
        minP: 0,
        presencePenalty: 1.5,
      },
    },
  });
});

test("also repairs the same auto-generated snapshot for Qwen3.5 and Qwen3.6", () => {
  for (const modelId of [
    "unsloth/Qwen3.5-9B-GGUF",
    "unsloth/Qwen3.6-27B-MTP-GGUF",
  ]) {
    const migrated = migrateLegacyQwenDefaults(
      settingsFor(modelId),
      modelId,
      true,
    );
    assert.equal(
      migrated.settings.inferenceParamsByModel?.[modelId]?.presencePenalty,
      1.5,
    );
  }
});

test("preserves an explicit partial presence override", () => {
  const settings = settingsFor(QWEN38, { presencePenalty: 0.0 });
  const migrated = migrateLegacyQwenDefaults(settings, QWEN38, true);

  assert.equal(migrated.patch, null);
  assert.equal(migrated.settings, settings);
});

test("preserves a customized snapshot", () => {
  const settings = settingsFor(QWEN38, {
    ...LEGACY_SNAPSHOT,
    temperature: 0.7,
  });
  const migrated = migrateLegacyQwenDefaults(settings, QWEN38, true);

  assert.equal(migrated.patch, null);
  assert.equal(migrated.settings, settings);
});

test("does not migrate generic Qwen3 or a custom preset", () => {
  const generic = settingsFor("unsloth/Qwen3-8B-GGUF");
  assert.equal(
    migrateLegacyQwenDefaults(generic, "unsloth/Qwen3-8B-GGUF", true).patch,
    null,
  );

  const custom = {
    ...settingsFor(QWEN38),
    activePreset: "Creative",
    activePresetSource: "custom" as const,
  };
  assert.equal(migrateLegacyQwenDefaults(custom, QWEN38, true).patch, null);
});

test("does not infer that globals belong to a newly active Qwen model", () => {
  const settings = settingsFor(QWEN38);
  const migrated = migrateLegacyQwenDefaults(settings, QWEN38, true, false);

  assert.equal(migrated.settings.inferenceParams?.presencePenalty, 0);
  assert.equal(migrated.patch?.inferenceParams, undefined);
});

test("migrates a global-only legacy install when ownership is established", () => {
  const settings = settingsFor(QWEN38);
  settings.inferenceParamsByModel = undefined;

  const migrated = migrateLegacyQwenDefaults(settings, QWEN38, false, true);

  assert.deepEqual(migrated.migratedModelIds, []);
  assert.equal(migrated.settings.inferenceParams?.temperature, 0.7);
  assert.equal(migrated.settings.inferenceParams?.topP, 0.8);
  assert.equal(migrated.settings.inferenceParams?.minP, 0);
  assert.equal(migrated.settings.inferenceParams?.presencePenalty, 1.5);
  assert.deepEqual(migrated.patch, {
    inferenceParams: {
      temperature: 0.7,
      topP: 0.8,
      minP: 0,
      presencePenalty: 1.5,
    },
  });
});

test("leaves global-only settings alone when active ownership is unknown", () => {
  const settings = settingsFor(QWEN38);
  settings.inferenceParamsByModel = undefined;

  assert.equal(
    migrateLegacyQwenDefaults(settings, QWEN38, false, false).patch,
    null,
  );
});

test("is idempotent after the legacy snapshot has been upgraded", () => {
  const first = migrateLegacyQwenDefaults(settingsFor(QWEN38), QWEN38, true);
  const second = migrateLegacyQwenDefaults(first.settings, QWEN38, true);

  assert.equal(second.patch, null);
  assert.deepEqual(second.migratedModelIds, []);
});
