// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PersistedChatSettings } from "../api/chat-settings-api";
import type { PersistedInferenceParams } from "../types/runtime";
import { resolveQwenThinkingParams } from "./qwen-sampling-table";

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

/**
 * The replacement table, read from the resolver model load and the Think toggle
 * use, so a corrected recommendation reaches persisted rows instead of this
 * file drifting into a second, stale source of truth.
 *
 * Null whenever the resolver gives no presencePenalty, which is exactly the
 * Qwen3.5/3.6/3.8 set: a generic Qwen3 row never had the bump, so it is not
 * ours to rewrite.
 */
function currentQwenDefaults(
  modelId: string,
  thinkingOn: boolean,
): PersistedInferenceParams | null {
  const resolved = resolveQwenThinkingParams(modelId, thinkingOn);
  if (resolved === null || resolved.presencePenalty === undefined) {
    return null;
  }
  return {
    temperature: resolved.temperature,
    topP: resolved.topP,
    topK: resolved.topK,
    minP: resolved.minP,
    presencePenalty: resolved.presencePenalty,
  };
}

export function isPresenceBumpQwen(modelId: string): boolean {
  // The resolver's own answer, not a second predicate: a model that gets the
  // bump on load is exactly the one whose stale row this repairs.
  return (
    resolveQwenThinkingParams(modelId, true)?.presencePenalty !== undefined
  );
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
        // Old global settings did not always serialize these two. A present
        // value must still match, so a customized snapshot is never mistaken
        // for generated legacy data.
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
  // Prefer an exact key when legacy storage holds two spellings; otherwise
  // normalize the sole case-insensitive match to the checkpoint spelling,
  // which is how replay indexes the map.
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
 * Upgrade the generic-Qwen snapshot Studio used to remember for Qwen3.5/3.6/3.8.
 * Matching every sampling field leaves a partial override untouched, including
 * a deliberate presencePenalty=0, and the context-derived maxTokens is kept.
 * Globals are eligible only when the caller can show they describe the active
 * checkpoint; a per-model map alone does not prove that.
 */
export function migrateLegacyQwenDefaults(
  settings: PersistedChatSettings,
  activeCheckpoint: string,
  thinkingOn: boolean,
  globalBelongsToActiveCheckpoint = false,
  migrateOwnedGlobalAlongsideModelMemory = false,
): QwenDefaultsMigration {
  // An empty map is no per-model memory, not memory that is empty. Both callers
  // normalize it away, but this is exported: leaving that to them makes a
  // global-only install fail to migrate depending on the path taken.
  const storedRaw = settings.inferenceParamsByModel;
  const stored =
    storedRaw !== undefined && Object.keys(storedRaw).length > 0
      ? storedRaw
      : undefined;
  if (!isBuiltInDefault(settings)) {
    return { settings, patch: null, migratedModelIds: [] };
  }

  const currentDefaults = currentQwenDefaults(activeCheckpoint, thinkingOn);
  if (currentDefaults === null) {
    return { settings, patch: null, migratedModelIds: [] };
  }
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
  const globalChanges = migrateGlobal
    ? changedDefaults(settings.inferenceParams ?? {}, currentGlobalDefaults)
    : null;
  // An already-current global produces {}, which is truthy and would be sent as
  // an empty conditional patch. Nothing to migrate is not a migration.
  const globalPatch =
    globalChanges && Object.keys(globalChanges).length > 0
      ? globalChanges
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
