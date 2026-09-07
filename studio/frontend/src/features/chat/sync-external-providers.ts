// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  reconcileLegacyProviderKeys,
  settleTasksIfCurrent,
} from "@/features/credentials/reconciliation";

import {
  type ProviderRegistryEntry,
  listProviderConfigs,
  listProviderRegistry,
  migrateProviderApiKey,
  updateProviderConfig,
} from "./api/providers-api";
import {
  CUSTOM_BACKEND_PROVIDER_TYPE,
  CUSTOM_PROVIDER_PRESETS,
  type ExternalProviderConfig,
  getExternalProviderApiKey,
  isCustomProviderType,
  isPromptCacheTtl,
  LEGACY_CUSTOM_PROVIDER_TYPE,
  pruneExternalProviderApiKeys,
  removeExternalProviderApiKey,

  PROVIDER_CAPABILITY_WILDCARD,
  pruneProviderModelCapabilities,
  getProviderModelCapabilities,
  setProviderModelCapabilities,
  supportsProviderPromptCaching,
  supportsProviderPromptCacheTtl,
  supportsProviderReasoningToggle,
} from "./external-providers";

const OPENAI_DEPRECATED_MODELS = new Set(["gpt-5.3"]);
// Rejected for every ChatGPT account, so drop it from selections saved earlier.
const OPENAI_CODEX_UNSUPPORTED_MODELS = new Set(["gpt-5.3-codex-spark"]);
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

export function mergeLearnedModelCapabilities(
  stored: Record<string, { vision?: boolean; studio_tools?: boolean }> | undefined,
  registryCapabilities: Record<string, { vision?: boolean; studio_tools?: boolean }> | undefined,
  supportsStudioTools: boolean | undefined,
): Record<string, { vision?: boolean; studio_tools?: boolean }> {
  const fromRegistry = registryCapabilities ?? {};
  // A plan-listed slug is learned at runtime and the registry cannot describe it, so rewriting this
  // map from the registry alone would drop it and leave the composer reading "unknown" as allowed
  // again on the next start.
  const capabilities: Record<string, { vision?: boolean; studio_tools?: boolean }> = {};
  for (const [modelId, capability] of Object.entries(stored ?? {})) {
    if (modelId !== PROVIDER_CAPABILITY_WILDCARD && !(modelId in fromRegistry)) {
      capabilities[modelId] = capability;
    }
  }
  Object.assign(capabilities, fromRegistry);
  if (typeof supportsStudioTools === "boolean") {
    capabilities[PROVIDER_CAPABILITY_WILDCARD] = {
      ...capabilities[PROVIDER_CAPABILITY_WILDCARD],
      studio_tools: supportsStudioTools,
    };
  }
  return capabilities;
}


export function pruneProviderModelIds(
  providerType: string,
  modelIds: string[],
): string[] {
  // Anthropic has no entry: a `-YYYYMMDD` id is the canonical name for the
  // whole pre-4.6 generation, not a snapshot. This mirrored the backend
  // denylist and outlived it, stripping those ids from the server catalog,
  // the seeds and saved selections alike.
  if (providerType === "openai") {
    return modelIds.filter((id) => !OPENAI_DEPRECATED_MODELS.has(id));
  }
  if (providerType === "openrouter") {
    return modelIds.filter((id) => !OPENROUTER_EXCLUDED_MODELS.has(id));
  }
  if (providerType === "openai_codex") {
    return modelIds.filter((id) => !OPENAI_CODEX_UNSUPPORTED_MODELS.has(id));
  }
  return modelIds;
}

/** Which model ids a synced connection ends up with: server, else browser-saved, else seed. The
 *  saved list is pruned BEFORE the emptiness test: a browser selection made up entirely of
 *  retired slugs is no selection at all, and resolving to it empties the picker and sends that
 *  empty list to a backend that rejects it. `serverModels` and `defaultModels` arrive pruned. */
export function resolveSyncedModelIds(
  providerType: string,
  serverModels: string[],
  savedModels: string[],
  defaultModels: string[],
): string[] {
  if (serverModels.length > 0) return serverModels;
  const prunedSaved = pruneProviderModelIds(providerType, savedModels);
  return prunedSaved.length > 0 ? prunedSaved : defaultModels;
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
  isCurrent?: () => boolean,
): Promise<ExternalProviderConfig[]> {
  const [registryRows, loadedConfigRows] = await Promise.all([
    listProviderRegistry(),
    listProviderConfigs(),
  ]);

  for (const entry of registryRows) {
    // Self-hosted model ids are user-supplied, so there is no per-model entry to key off. The
    // registry declares studio_tools once per provider type; park it under the wildcard so the
    // per-model lookup can fall back to it.
    const capabilities = mergeLearnedModelCapabilities(
      getProviderModelCapabilities(entry.provider_type),
      entry.model_capabilities,
      entry.supports_studio_tools,
    );
    setProviderModelCapabilities(entry.provider_type, capabilities);
  }
  // Writing per returned entry can only correct what came back. Capabilities are persisted in
  // localStorage and outlive the backend that wrote them, so a provider the registry has stopped
  // listing would otherwise keep its last `studio_tools: true` forever.
  pruneProviderModelCapabilities(registryRows.map((entry) => entry.provider_type));
  const configRows = await reconcileLegacyProviderKeys(loadedConfigRows, {
    getLegacyKey: getExternalProviderApiKey,
    saveLegacyKey: migrateProviderApiKey,
    removeLegacyKey: removeExternalProviderApiKey,

    isCurrent,
  });

  if (isCurrent && !isCurrent()) return existingProviders;
  pruneExternalProviderApiKeys(loadedConfigRows.map((config) => config.id));

  const existingById = new Map<string, ExternalProviderConfig>();
  for (const provider of existingProviders) {
    existingById.set(provider.id, provider);
  }

  const backfillTasks: Array<() => Promise<unknown>> = [];
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
      const resolvedModels = resolveSyncedModelIds(
        uiProviderType,
        serverModels,
        savedModels,
        defaultModels,
      );
      const resolvedAvailableModels = resolveSyncedModelIds(
        uiProviderType,
        serverAvailableModels,
        savedAvailableModels,
        defaultModels,
      );
      const needsModelBackfill =
        serverModels.length === 0 && savedModels.length > 0;
      const needsAvailableBackfill =
        serverAvailableModels.length === 0 && savedAvailableModels.length > 0;
      if (needsModelBackfill || needsAvailableBackfill) {
        backfillTasks.push(() =>
          updateProviderConfig(config.id, {
            models: resolvedModels,
            availableModels: resolvedAvailableModels,
          }),
        );
      }
      const synced: ExternalProviderConfig = {
        id: config.id,
        providerType: uiProviderType,
        // Beside the UI type, which disagrees for a legacy row saved as `openai`: only the stored type
        // decides what the backend accepts.
        backendProviderType: config.provider_type,
        name: config.display_name,
        baseUrl: config.base_url ?? "",
        models: resolvedModels,
        availableModels: resolvedAvailableModels,
        maxOutputTokens: config.max_output_tokens ?? undefined,

        hasApiKey: config.has_api_key,

        authKind: config.auth_kind,
        authStatus: config.auth_status,
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

  if (isCurrent && !isCurrent()) return existingProviders;

  await settleTasksIfCurrent(backfillTasks, isCurrent);
  return syncedProviders;
}
