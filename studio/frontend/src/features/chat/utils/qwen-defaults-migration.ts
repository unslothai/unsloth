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
} as const;

const LEGACY_GLOBAL_QWEN_DEFAULTS = {
  temperature: 0.6,
  topP: 0.95,
  minP: 0.01,
  presencePenalty: 0.0,
} as const;

const LEGACY_OPTIONAL_GLOBAL_QWEN_DEFAULTS = {
  topK: 20,
  repetitionPenalty: 1.0,
} as const;

const CURRENT_QWEN_THINKING_DEFAULTS = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.0,
  presencePenalty: 1.5,
} as const satisfies PersistedInferenceParams;

const CURRENT_QWEN_NON_THINKING_DEFAULTS = {
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  minP: 0.0,
  presencePenalty: 1.5,
} as const satisfies PersistedInferenceParams;

export function isPresenceBumpQwen(modelId: string): boolean {
  return /(?:^|[^a-z0-9])qwen3\.(?:5|6|8)(?:$|[-_/\\])/i.test(modelId);
}

function isLegacyQwenDefaultSnapshot(
  params: PersistedInferenceParams,
): boolean {
  return Object.entries(LEGACY_QWEN_DEFAULTS).every(
    ([key, value]) => params[key as keyof PersistedInferenceParams] === value,
  );
}

function isLegacyGlobalQwenDefaultSnapshot(
  params: PersistedInferenceParams | undefined,
): boolean {
  return (
    params !== undefined &&
    Object.entries(LEGACY_GLOBAL_QWEN_DEFAULTS).every(
      ([key, value]) => params[key as keyof PersistedInferenceParams] === value,
    ) &&
    Object.entries(LEGACY_OPTIONAL_GLOBAL_QWEN_DEFAULTS).every(
      ([key, value]) => {
        const stored = params[key as keyof PersistedInferenceParams];
        // Old global settings did not always serialize these two unchanged
        // defaults. A present value must still match, so customized snapshots
        // cannot be mistaken for generated legacy data.
        return stored === undefined || stored === value;
      },
    )
  );
}

function changedDefaults(
  legacy: PersistedInferenceParams,
  current: PersistedInferenceParams,
): PersistedInferenceParams {
  const changed: PersistedInferenceParams = {};
  for (const [key, value] of Object.entries(current)) {
    const field = key as keyof PersistedInferenceParams;
    if (legacy[field] !== value) {
      (changed as Record<string, unknown>)[key] = value;
    }
  }
  return changed;
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

function migrateStoredModelDefaults(
  stored: Record<string, PersistedInferenceParams> | undefined,
  activeCheckpoint: string,
  currentDefaults: PersistedInferenceParams,
): {
  migratedModelIds: string[];
  migratedByModel: Record<string, PersistedInferenceParams>;
  patchByModel: Record<string, PersistedInferenceParams>;
} {
  const migratedModelIds: string[] = [];
  const migratedByModel: Record<string, PersistedInferenceParams> = {};
  const patchByModel: Record<string, PersistedInferenceParams> = {};
  const storedEntries = Object.entries(stored ?? {});
  // Prefer an exact key when malformed/legacy storage contains two spellings.
  // Otherwise normalize the sole case-insensitive match to the checkpoint
  // spelling used by replay, which indexes the map by exact key.
  const activeStoredId =
    storedEntries.find(([modelId]) => modelId === activeCheckpoint)?.[0] ??
    storedEntries.find(
      ([modelId]) => modelId.toLowerCase() === activeCheckpoint.toLowerCase(),
    )?.[0];
  for (const [modelId, entry] of storedEntries) {
    const isActiveCheckpoint = modelId === activeStoredId;
    if (
      isActiveCheckpoint &&
      isPresenceBumpQwen(modelId) &&
      isLegacyQwenDefaultSnapshot(entry)
    ) {
      const normalizedModelId = activeCheckpoint;
      const migratedEntry = { ...entry, ...currentDefaults };
      migratedModelIds.push(normalizedModelId);
      migratedByModel[normalizedModelId] = migratedEntry;
      patchByModel[normalizedModelId] =
        modelId === normalizedModelId
          ? changedDefaults(entry, currentDefaults)
          : migratedEntry;
    } else {
      migratedByModel[modelId] = entry;
    }
  }
  return { migratedModelIds, migratedByModel, patchByModel };
}

/**
 * Upgrade the complete generic-Qwen snapshot that Studio used to remember for
 * Qwen3.5/3.6/3.8. Matching every sampling field keeps an explicit partial
 * override (including a deliberate presencePenalty=0) untouched, while the
 * context-derived maxTokens budget is deliberately preserved. Globals are
 * eligible only when the caller can establish that they describe the active
 * checkpoint; a per-model map alone is not proof of that ownership.
 */
export function migrateLegacyQwenDefaults(
  settings: PersistedChatSettings,
  activeCheckpoint: string,
  thinkingOn: boolean,
  globalBelongsToActiveCheckpoint = false,
  migrateOwnedGlobalAlongsideModelMemory = false,
): QwenDefaultsMigration {
  const stored = settings.inferenceParamsByModel;
  if (!isBuiltInDefault(settings)) {
    return { settings, patch: null, migratedModelIds: [] };
  }

  const currentDefaults = thinkingOn
    ? CURRENT_QWEN_THINKING_DEFAULTS
    : CURRENT_QWEN_NON_THINKING_DEFAULTS;
  const currentGlobalDefaults: PersistedInferenceParams = {
    temperature: currentDefaults.temperature,
    topP: currentDefaults.topP,
    minP: currentDefaults.minP,
    presencePenalty: currentDefaults.presencePenalty,
  };

  const { migratedModelIds, migratedByModel, patchByModel } =
    migrateStoredModelDefaults(stored, activeCheckpoint, currentDefaults);

  const migrateGlobal =
    (stored === undefined || migrateOwnedGlobalAlongsideModelMemory) &&
    globalBelongsToActiveCheckpoint &&
    isPresenceBumpQwen(activeCheckpoint) &&
    isLegacyGlobalQwenDefaultSnapshot(settings.inferenceParams);
  const globalPatch = migrateGlobal
    ? changedDefaults(settings.inferenceParams ?? {}, currentGlobalDefaults)
    : null;

  if (migratedModelIds.length === 0 && !globalPatch) {
    return { settings, patch: null, migratedModelIds };
  }

  return {
    settings: {
      ...settings,
      ...(stored ? { inferenceParamsByModel: migratedByModel } : {}),
      ...(migrateGlobal
        ? {
            inferenceParams: {
              ...settings.inferenceParams,
              ...currentGlobalDefaults,
            },
          }
        : {}),
    },
    patch: {
      ...(migratedModelIds.length > 0
        ? { inferenceParamsByModel: patchByModel }
        : {}),
      ...(globalPatch ? { inferenceParams: globalPatch } : {}),
    },
    migratedModelIds,
  };
}
