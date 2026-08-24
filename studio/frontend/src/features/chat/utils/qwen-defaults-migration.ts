// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PersistedChatSettings } from "../api/chat-settings-api";
import type { PersistedInferenceParams } from "../types/runtime";

const LEGACY_QWEN_DEFAULTS = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.01,
  repetitionPenalty: 1.0,
  presencePenalty: 0.0,
  maxTokens: 8192,
} as const;

const CURRENT_QWEN_DEFAULT_PATCH = {
  minP: 0.0,
  presencePenalty: 1.5,
} as const satisfies PersistedInferenceParams;

function isPresenceBumpQwen(modelId: string): boolean {
  const normalized = modelId.toLowerCase();
  return (
    normalized.includes("qwen3.5") ||
    normalized.includes("qwen3.6") ||
    normalized.includes("qwen3.8")
  );
}

function isLegacyQwenDefaultSnapshot(
  params: PersistedInferenceParams,
): boolean {
  return Object.entries(LEGACY_QWEN_DEFAULTS).every(
    ([key, value]) => params[key as keyof PersistedInferenceParams] === value,
  );
}

function isBuiltInDefault(settings: PersistedChatSettings): boolean {
  return (
    (settings.activePreset === undefined ||
      settings.activePreset === "Default") &&
    (settings.activePresetSource === undefined ||
      settings.activePresetSource === "builtin-default")
  );
}

export type QwenDefaultsMigration = {
  settings: PersistedChatSettings;
  patch: PersistedChatSettings | null;
  migratedModelIds: string[];
};

/**
 * Upgrade the complete generic-Qwen snapshot that Studio used to remember for
 * Qwen3.5/3.6/3.8. Matching every sampling field keeps an explicit partial
 * override (including a deliberate presencePenalty=0) untouched.
 */
export function migrateLegacyQwenDefaults(
  settings: PersistedChatSettings,
  activeCheckpoint: string,
): QwenDefaultsMigration {
  const stored = settings.inferenceParamsByModel;
  if (!stored) {
    return { settings, patch: null, migratedModelIds: [] };
  }
  if (!isBuiltInDefault(settings)) {
    return { settings, patch: null, migratedModelIds: [] };
  }

  const migratedModelIds: string[] = [];
  const migratedByModel: Record<string, PersistedInferenceParams> = {};
  const patchByModel: Record<string, PersistedInferenceParams> = {};

  for (const [modelId, entry] of Object.entries(stored)) {
    if (isPresenceBumpQwen(modelId) && isLegacyQwenDefaultSnapshot(entry)) {
      migratedModelIds.push(modelId);
      migratedByModel[modelId] = { ...entry, ...CURRENT_QWEN_DEFAULT_PATCH };
      patchByModel[modelId] = CURRENT_QWEN_DEFAULT_PATCH;
    } else {
      migratedByModel[modelId] = entry;
    }
  }

  if (migratedModelIds.length === 0) {
    return { settings, patch: null, migratedModelIds };
  }

  const activeWasMigrated = migratedModelIds.some(
    (modelId) => modelId.toLowerCase() === activeCheckpoint.toLowerCase(),
  );
  const migrateGlobal =
    activeWasMigrated &&
    settings.inferenceParams?.minP === LEGACY_QWEN_DEFAULTS.minP &&
    settings.inferenceParams?.presencePenalty ===
      LEGACY_QWEN_DEFAULTS.presencePenalty;

  return {
    settings: {
      ...settings,
      inferenceParamsByModel: migratedByModel,
      ...(migrateGlobal
        ? {
            inferenceParams: {
              ...settings.inferenceParams,
              ...CURRENT_QWEN_DEFAULT_PATCH,
            },
          }
        : {}),
    },
    patch: {
      inferenceParamsByModel: patchByModel,
      ...(migrateGlobal ? { inferenceParams: CURRENT_QWEN_DEFAULT_PATCH } : {}),
    },
    migratedModelIds,
  };
}
