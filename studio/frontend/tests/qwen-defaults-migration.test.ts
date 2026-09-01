// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

// The migration now reads the shared sampling table by extensionless specifier,
// like the rest of src/, so this test needs the bundler resolver too.
register("./bundler-resolver.mjs", import.meta.url);

import type { PersistedChatSettings } from "../src/features/chat/api/chat-settings-api.ts";
const { migrateLegacyQwenDefaults, isPresenceBumpQwen } = await import(
  "../src/features/chat/utils/qwen-defaults-migration.ts"
);

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

test("preserves context-derived token budgets while migrating sampling", () => {
  for (const maxTokens of [4096, 32768]) {
    const settings = settingsFor(QWEN38, {
      ...LEGACY_SNAPSHOT,
      maxTokens,
    });
    const migrated = migrateLegacyQwenDefaults(
      settings,
      QWEN38,
      true,
      true,
    );

    assert.equal(
      migrated.settings.inferenceParamsByModel?.[QWEN38]?.maxTokens,
      maxTokens,
    );
    assert.deepEqual(migrated.patch?.inferenceParamsByModel?.[QWEN38], {
      minP: 0,
      presencePenalty: 1.5,
    });
  }
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

test("migrates only the active model when several legacy Qwen rows are saved", () => {
  const qwen36Small = "unsloth/Qwen3.6-9B-GGUF";
  const settings = settingsFor(QWEN38);
  settings.inferenceParamsByModel = {
    [QWEN38]: LEGACY_SNAPSHOT,
    [qwen36Small]: LEGACY_SNAPSHOT,
  };

  const migrated = migrateLegacyQwenDefaults(settings, QWEN38, true);

  assert.deepEqual(migrated.migratedModelIds, [QWEN38]);
  assert.equal(
    migrated.settings.inferenceParamsByModel?.[QWEN38]?.presencePenalty,
    1.5,
  );
  assert.deepEqual(
    migrated.settings.inferenceParamsByModel?.[qwen36Small],
    LEGACY_SNAPSHOT,
  );
  assert.equal(
    migrated.patch?.inferenceParamsByModel?.[qwen36Small],
    undefined,
  );
});

test("normalizes a case-insensitive saved key to the active checkpoint", () => {
  const lowerCaseKey = QWEN38.toLowerCase();
  const migrated = migrateLegacyQwenDefaults(
    settingsFor(lowerCaseKey),
    QWEN38,
    true,
  );

  assert.deepEqual(migrated.migratedModelIds, [QWEN38]);
  assert.equal(
    migrated.settings.inferenceParamsByModel?.[QWEN38]?.presencePenalty,
    1.5,
  );
  assert.equal(
    migrated.settings.inferenceParamsByModel?.[lowerCaseKey],
    undefined,
  );
  assert.deepEqual(
    migrated.patch?.inferenceParamsByModel?.[QWEN38],
    {
      ...LEGACY_SNAPSHOT,
      minP: 0,
      presencePenalty: 1.5,
    },
  );
});

test("does not migrate dormant Qwen rows while a non-Qwen model is active", () => {
  const settings = settingsFor(QWEN38);

  const migrated = migrateLegacyQwenDefaults(
    settings,
    "unsloth/Llama-3.2-3B-Instruct-GGUF",
    true,
  );

  assert.equal(migrated.patch, null);
  assert.equal(migrated.settings, settings);
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

test("external checkpoints are decoded before the family is matched", () => {
  // buildExternalModelId percent-encodes, so a provider-namespaced Qwen arrives
  // as external::<provider>::Qwen%2FQwen3.8-27B. Matching that raw would put an
  // alphanumeric "f" against the family segment and drop the presence bump.
  const encoded = `external::openrouter::${encodeURIComponent("Qwen/Qwen3.8-27B")}`;
  assert.equal(isPresenceBumpQwen(encoded), true);
  assert.equal(
    isPresenceBumpQwen(
      `external::openrouter::${encodeURIComponent("Qwen/Qwen3-8B")}`,
    ),
    false,
  );

  const settings = settingsFor(encoded);
  const migrated = migrateLegacyQwenDefaults(settings, encoded, true);
  assert.equal(
    migrated.patch?.inferenceParamsByModel?.[encoded]?.presencePenalty,
    1.5,
  );
});

test("matches presence-bump versions only at model-id boundaries", () => {
  const future = settingsFor("unsloth/Qwen3.80-27B-GGUF");
  assert.equal(
    migrateLegacyQwenDefaults(future, "unsloth/Qwen3.80-27B-GGUF", true).patch,
    null,
  );
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

test("preserves a global-only install's context-derived token budget", () => {
  const settings = settingsFor(QWEN38);
  settings.inferenceParamsByModel = undefined;
  settings.inferenceParams = {
    ...settings.inferenceParams,
    maxTokens: 32768,
  };

  const migrated = migrateLegacyQwenDefaults(settings, QWEN38, true, true);

  assert.equal(migrated.settings.inferenceParams?.maxTokens, 32768);
  assert.equal(migrated.patch?.inferenceParams?.maxTokens, undefined);
  assert.equal(migrated.patch?.inferenceParams?.presencePenalty, 1.5);
});

test("preserves customized optional fields in a global-only snapshot", () => {
  for (const override of [{ topK: 40 }, { repetitionPenalty: 1.1 }]) {
    const settings = settingsFor(QWEN38);
    settings.inferenceParamsByModel = undefined;
    settings.inferenceParams = {
      ...settings.inferenceParams,
      ...override,
    };

    const migrated = migrateLegacyQwenDefaults(settings, QWEN38, false, true);

    assert.equal(migrated.patch, null);
    assert.equal(migrated.settings, settings);
  }
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

test("an Ollama manifest reference is decoded before the family is matched", () => {
  // The inventory ref keeps its quote(safe='') encoding all the way into
  // inference status, so every separator arrives as %2F.
  const ref = `ollama-manifest:${encodeURIComponent(
    "/home/u/.ollama/manifests/registry.ollama.ai/library/qwen3.8/latest",
  )}`;
  assert.equal(isPresenceBumpQwen(ref), true);
  assert.equal(
    isPresenceBumpQwen(
      `ollama-manifest:${encodeURIComponent(
        "/home/u/.ollama/manifests/registry.ollama.ai/library/qwen3/latest",
      )}`,
    ),
    false,
  );
  // A malformed escape must not lose the checkpoint entirely.
  assert.equal(isPresenceBumpQwen("ollama-manifest:%E0%A4%A/qwen3.8/latest"), true);
});

test("POSIX paths differing only by case are not the same model", () => {
  const active = "/home/u/Models/qwen3.8-27b";
  const other = "/home/u/models/qwen3.8-27b";
  const settings = settingsFor(other);

  const migrated = migrateLegacyQwenDefaults(settings, active, true);

  // The other path is a different file on a case-sensitive filesystem, so its
  // row must not be moved under the active checkpoint.
  assert.deepEqual(migrated.migratedModelIds, []);
  assert.equal(migrated.patch, null);
});

test("a Windows path still folds case", () => {
  const active = "C:\\Models\\qwen3.8-27b";
  const settings = settingsFor("c:\\models\\qwen3.8-27b");

  const migrated = migrateLegacyQwenDefaults(settings, active, true);

  assert.deepEqual(migrated.migratedModelIds, [active]);
});
