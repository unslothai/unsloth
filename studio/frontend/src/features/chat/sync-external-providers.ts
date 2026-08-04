// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ProviderRegistryEntry,
  listProviderConfigs,
  listProviderRegistry,
  updateProviderConfig,
} from "./api/providers-api";
import {
  CUSTOM_BACKEND_PROVIDER_TYPE,
  CUSTOM_PROVIDER_PRESETS,
  type ExternalProviderConfig,
  isCustomProviderType,
  isPromptCacheTtl,
  LEGACY_CUSTOM_PROVIDER_TYPE,
  supportsProviderPromptCaching,
  supportsProviderPromptCacheTtl,
  supportsProviderReasoningToggle,
} from "./external-providers";

const ANTHROPIC_DATED_SNAPSHOT_SUFFIX = /-\d{8}$/;
const OPENAI_DEPRECATED_MODELS = new Set(["gpt-5.3"]);
const OPENROUTER_EXCLUDED_MODELS = new Set([
  "google/chirp-3",
  "kwaivgi/kling-v3.0-pro",
  "openai/whisper-1",
  "openai/gpt-4o-mini-transcribe",
  "recraft/recraft-v4-pro",
]);

function normalizeUrl(input: string): string {
  return input.trim().replace(/\/+$/, "");
}

export function resolveUiProviderTypeFromConfig(
  configProviderType: string,
  configDisplayName: string | null | undefined,
  configBaseUrl: string | null | undefined,
  registryRows: ProviderRegistryEntry[],
  existingProviderType: string | undefined,
): string {
  if (existingProviderType && isCustomProviderType(existingProviderType)) {
    return existingProviderType;
  }
  if (configProviderType !== CUSTOM_BACKEND_PROVIDER_TYPE) {
    return configProviderType;
  }
  const displayName = (configDisplayName ?? "").trim().toLowerCase();
  const matchingCustomPreset = CUSTOM_PROVIDER_PRESETS.find(
    (preset) => preset.displayName.toLowerCase() === displayName,
  );
  if (matchingCustomPreset) {
    return matchingCustomPreset.providerType;
  }
  const openAiRegistry = registryRows.find(
    (entry) => entry.provider_type === CUSTOM_BACKEND_PROVIDER_TYPE,
  );
  if (!openAiRegistry) {
    return configProviderType;
  }
  const openAiDisplayName = openAiRegistry.display_name.trim().toLowerCase();
  if (displayName.length > 0 && displayName !== openAiDisplayName) {
    return LEGACY_CUSTOM_PROVIDER_TYPE;
  }
  const configUrl = normalizeUrl(configBaseUrl ?? "");
  const defaultUrl = normalizeUrl(openAiRegistry.base_url ?? "");
  if (configUrl.length > 0 && configUrl !== defaultUrl) {
    return LEGACY_CUSTOM_PROVIDER_TYPE;
  }
  return configProviderType;
}

export function pruneProviderModelIds(
  providerType: string,
  modelIds: string[],
): string[] {
  if (providerType === "anthropic") {
    return modelIds.filter((id) => !ANTHROPIC_DATED_SNAPSHOT_SUFFIX.test(id));
  }
  if (providerType === "openai") {
    return modelIds.filter((id) => !OPENAI_DEPRECATED_MODELS.has(id));
  }
  if (providerType === "openrouter") {
    return modelIds.filter((id) => !OPENROUTER_EXCLUDED_MODELS.has(id));
  }
  return modelIds;
}

/** Carry browser-local provider knobs through a backend sync rebuild. */
export function mergeLocalProviderOptions(
  existing: ExternalProviderConfig | undefined,
  synced: ExternalProviderConfig,
): ExternalProviderConfig {
  if (!existing) {
    return synced;
  }
  const providerType = synced.providerType;
  return {
    ...synced,
    enablePromptCaching: supportsProviderPromptCaching(providerType)
      ? (existing.enablePromptCaching ?? synced.enablePromptCaching ?? true)
      : undefined,
    promptCacheTtl:
      supportsProviderPromptCacheTtl(providerType) &&
      isPromptCacheTtl(existing.promptCacheTtl)
        ? existing.promptCacheTtl
        : synced.promptCacheTtl,
    isReasoningModel: supportsProviderReasoningToggle(providerType)
      ? (existing.isReasoningModel ?? synced.isReasoningModel)
      : undefined,
    openaiContainerTtlMinutes:
      providerType === "openai" &&
      typeof existing.openaiContainerTtlMinutes === "number" &&
      existing.openaiContainerTtlMinutes >= 1
        ? Math.min(existing.openaiContainerTtlMinutes, 20)
        : synced.openaiContainerTtlMinutes,
  };
}

/** Merge enabled backend provider configs with local store state. */
export async function syncExternalProvidersFromBackend(
  existingProviders: ExternalProviderConfig[],
): Promise<ExternalProviderConfig[]> {
  const [registryRows, configRows] = await Promise.all([
    listProviderRegistry(),
    listProviderConfigs(),
  ]);

  const existingById = new Map<string, ExternalProviderConfig>();
  for (const provider of existingProviders) {
    existingById.set(provider.id, provider);
  }

  const backfillTasks: Promise<unknown>[] = [];
  const syncedProviders = configRows
    .filter((config) => config.is_enabled)
    .map((config) => {
      const existing = existingById.get(config.id);
      const uiProviderType = resolveUiProviderTypeFromConfig(
        config.provider_type,
        config.display_name,
        config.base_url,
        registryRows,
        existing?.providerType,
      );
      const createdAt = Number.isFinite(Date.parse(config.created_at))
        ? Date.parse(config.created_at)
        : Date.now();
      const updatedAt = Number.isFinite(Date.parse(config.updated_at))
        ? Date.parse(config.updated_at)
        : Date.now();
      const registryEntry =
        registryRows.find((entry) => entry.provider_type === uiProviderType) ??
        registryRows.find((entry) => entry.provider_type === config.provider_type);
      const defaultModels = pruneProviderModelIds(
        uiProviderType,
        registryEntry?.default_models ?? [],
      );
      const serverModels = pruneProviderModelIds(
        uiProviderType,
        config.models ?? [],
      );
      const serverAvailableModels = pruneProviderModelIds(
        uiProviderType,
        config.available_models ?? [],
      );
      const savedModels = existing?.models ?? [];
      const savedAvailableModels = existing?.availableModels ?? [];
      const resolvedModels = pruneProviderModelIds(
        uiProviderType,
        serverModels.length > 0
          ? serverModels
          : savedModels.length > 0
            ? savedModels
            : defaultModels,
      );
      const resolvedAvailableModels = pruneProviderModelIds(
        uiProviderType,
        serverAvailableModels.length > 0
          ? serverAvailableModels
          : savedAvailableModels.length > 0
            ? savedAvailableModels
            : defaultModels,
      );
      const needsModelBackfill =
        serverModels.length === 0 && savedModels.length > 0;
      const needsAvailableBackfill =
        serverAvailableModels.length === 0 && savedAvailableModels.length > 0;
      if (needsModelBackfill || needsAvailableBackfill) {
        backfillTasks.push(
          updateProviderConfig(config.id, {
            models: resolvedModels,
            availableModels: resolvedAvailableModels,
          }),
        );
      }
      const synced: ExternalProviderConfig = {
        id: config.id,
        providerType: uiProviderType,
        name: config.display_name,
        baseUrl: config.base_url ?? "",
        models: resolvedModels,
        availableModels: resolvedAvailableModels,
        enablePromptCaching: supportsProviderPromptCaching(uiProviderType)
          ? (existing?.enablePromptCaching ?? true)
          : undefined,
        isReasoningModel: supportsProviderReasoningToggle(uiProviderType)
          ? existing?.isReasoningModel === true
          : undefined,
        createdAt: existing?.createdAt ?? createdAt,
        updatedAt,
      };
      return mergeLocalProviderOptions(existing, synced);
    });

  if (backfillTasks.length > 0) {
    await Promise.allSettled(backfillTasks);
  }
  return syncedProviders;
}
