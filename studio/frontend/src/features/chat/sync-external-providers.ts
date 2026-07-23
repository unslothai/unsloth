// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type ProviderRegistryEntry,
  listProviderConfigs,
  listProviderRegistry,
} from "./api/providers-api";
import type { ExternalProviderConfig } from "./external-providers";
import {
  CUSTOM_BACKEND_PROVIDER_TYPE,
  CUSTOM_PROVIDER_PRESETS,
  isCustomProviderType,
  LEGACY_CUSTOM_PROVIDER_TYPE,
  supportsProviderPromptCaching,
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

  return configRows
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
      const existingModels = pruneProviderModelIds(
        uiProviderType,
        serverModels.length > 0
          ? serverModels
          : savedModels.length > 0
            ? savedModels
            : defaultModels,
      );
      const existingAvailableModels = pruneProviderModelIds(
        uiProviderType,
        serverAvailableModels.length > 0
          ? serverAvailableModels
          : savedAvailableModels.length > 0
            ? savedAvailableModels
            : defaultModels,
      );
      return {
        id: config.id,
        providerType: uiProviderType,
        name: config.display_name,
        baseUrl: config.base_url ?? "",
        models: existingModels,
        availableModels: existingAvailableModels,
        enablePromptCaching: supportsProviderPromptCaching(uiProviderType)
          ? (existing?.enablePromptCaching ?? true)
          : undefined,
        isReasoningModel: supportsProviderReasoningToggle(uiProviderType)
          ? existing?.isReasoningModel === true
          : undefined,
        createdAt: existing?.createdAt ?? createdAt,
        updatedAt,
      };
    });
}
