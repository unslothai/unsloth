// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PersistedChatSettings } from "../api/chat-settings-api";
import type { PersistedInferenceParams } from "../types/runtime";
import { isExternalModelId } from "../external-providers";
import { normalizeModelIdentity } from "../../hub/lib/model-identity";
import {
  isOllamaManifestRef,
  resolveQwenThinkingParams,
} from "./qwen-sampling-table";

function isOpaqueModelRef(modelId: string): boolean {
  return isExternalModelId(modelId) || isOllamaManifestRef(modelId);
}

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

// Exact output of the immediately preceding migration for Qwen3.8 thinking.
// Matching every sampling field keeps user-tuned rows out of this migration.
const PREVIOUS_QWEN38_THINKING_DEFAULTS = {
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  minP: 0.0,
  repetitionPenalty: 1.0,
  presencePenalty: 1.5,
} as const;

const PREVIOUS_QWEN38_THINKING_GLOBAL_DEFAULTS = {
  temperature: 0.6,
  topP: 0.95,
  minP: 0.0,
  presencePenalty: 1.5,
} as const;

/**
 * The replacement table, read from the same resolver load and the Think toggle
 * use, so this file cannot drift into a second source of truth. Null whenever
 * the resolver gives no presencePenalty: a generic Qwen3 row never had the bump,
 * so it is not ours to rewrite.
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

function matchesDefaults(
  params: PersistedInferenceParams,
  defaults: Partial<PersistedInferenceParams>,
): boolean {
  return Object.entries(defaults).every(
    ([key, value]) => params[key as keyof PersistedInferenceParams] === value,
  );
}

function isLegacyQwenDefaultSnapshot(
  params: PersistedInferenceParams,
  currentDefaults: PersistedInferenceParams,
): boolean {
  // The shared resolver returns zero presence only for Qwen3.8 thinking.
  return (
    matchesDefaults(params, LEGACY_QWEN_DEFAULTS) ||
    (currentDefaults.presencePenalty === 0 &&
      matchesDefaults(params, PREVIOUS_QWEN38_THINKING_DEFAULTS))
  );
}

function isLegacyGlobalQwenDefaultSnapshot(
  params: PersistedInferenceParams | undefined,
  currentDefaults: PersistedInferenceParams,
): boolean {
  return (
    params !== undefined &&
    (matchesDefaults(params, LEGACY_GLOBAL_QWEN_DEFAULTS) ||
      (currentDefaults.presencePenalty === 0 &&
        matchesDefaults(params, PREVIOUS_QWEN38_THINKING_GLOBAL_DEFAULTS))) &&
    Object.entries(LEGACY_OPTIONAL_GLOBAL_QWEN_DEFAULTS).every(
      ([key, value]) => {
        const stored = params[key as keyof PersistedInferenceParams];
        // Old globals did not always serialize these two, but a present value
        // must still match, or a customization reads as generated data.
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
  // normalize the sole case-insensitive match to the checkpoint spelling, which
  // is how replay indexes the map. normalizeModelIdentity, not toLowerCase: it
  // preserves case for POSIX paths, which can name two different files. Opaque
  // ids compare exactly and have no legacy spellings to reconcile.
  const activeIdentity = normalizeModelIdentity(activeCheckpoint);
  const activeIsOpaque = isOpaqueModelRef(activeCheckpoint);
  // Only the sole alias, checked rather than assumed: with two spellings one
  // may hold the legacy snapshot and the other the user's own sampling, and
  // insertion order would serve generated defaults over a customization.
  const aliases = activeIsOpaque
    ? []
    : storedEntries.filter(
        ([modelId]) =>
          !isOpaqueModelRef(modelId) &&
          normalizeModelIdentity(modelId) === activeIdentity,
      );
  const activeStoredId =
    storedEntries.find(([modelId]) => modelId === activeCheckpoint)?.[0] ??
    (aliases.length === 1 ? aliases[0][0] : undefined);
  for (const [modelId, entry] of storedEntries) {
    const isActiveCheckpoint = modelId === activeStoredId;
    if (
      isActiveCheckpoint &&
      isPresenceBumpQwen(modelId) &&
      isLegacyQwenDefaultSnapshot(entry, currentDefaults)
    ) {
      const normalizedModelId = activeCheckpoint;
      const migratedEntry = { ...entry, ...currentDefaults };
      migratedModelIds.push(normalizedModelId);
      migratedByModel[normalizedModelId] = migratedEntry;
      patchByModel[normalizedModelId] =
        modelId === normalizedModelId
          ? changedDefaults(entry, currentDefaults)
          : migratedEntry;
      if (modelId !== normalizedModelId) {
        // The server merge only sets keys, so the spelling normalized away
        // survives and a later status naming it would replay the stale row.
        migratedByModel[modelId] = migratedEntry;
        patchByModel[modelId] = changedDefaults(entry, currentDefaults);
      }
    } else {
      migratedByModel[modelId] = entry;
    }
  }
  return { migratedModelIds, migratedByModel, patchByModel };
}

/**
 * Upgrade the generated Qwen snapshots Studio used to remember for Qwen3.5/3.6/3.8.
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
  // An empty map is no per-model memory, not memory that is empty. Normalized
  // here, not in the callers: this is exported.
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
    isLegacyGlobalQwenDefaultSnapshot(
      settings.inferenceParams,
      currentDefaults,
    );
  const globalChanges = migrateGlobal
    ? changedDefaults(settings.inferenceParams ?? {}, currentGlobalDefaults)
    : null;
  // An already-current global produces {}, which is truthy and would be sent as
  // an empty patch.
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
