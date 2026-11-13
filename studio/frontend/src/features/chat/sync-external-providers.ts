// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  reconcileLegacyProviderKeys,
  settleTasksIfCurrent,
} from "@/features/credentials/reconciliation";

import {
  type CodexSubscriptionModels,
  type ModelCatalogResponse,
  type ProviderRegistryEntry,
  fetchCodexSubscriptionModels,
  fetchModelCatalog,
  listProviderConfigs,
  listProviderModelCapabilities,
  listProviderModels,
  listProviderRegistry,
  migrateProviderApiKey,
  updateProviderConfig,
} from "./api/providers-api";
import { normalizeCustomReasoningConfig } from "./custom-reasoning";
import {
  modelsDevCatalogFetchedAt,
  providerModelCatalogFetchedAt,
  setModelsDevCatalog,
  setProviderModelCatalog,
} from "./model-catalog";
import { learnModelDisplayNames } from "./model-display-names";
import {
  CUSTOM_BACKEND_PROVIDER_TYPE,
  CUSTOM_PROVIDER_DISPLAY_NAME,
  CUSTOM_PROVIDER_PRESETS,
  connectionApiFields,
  type ExternalProviderConfig,
  extraHeadersFromSaved,
  getExternalProviderApiKey,
  isCustomProviderType,
  isPromptCacheTtl,
  learnCatalogModelCapabilities,
  loadNewModelChecksAt,
  newModelCheckOf,
  type NewModelCheckInterval,
  saveNewModelCheckAt,
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
  supportsRemoteModelCatalog,
  toExternalBackendProviderType,
} from "./external-providers";
import {
  providerSavesInFlight,
  useExternalProvidersStore,
} from "./stores/external-providers-store";

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

function parseHttpEndpointHost(
  input: string | null | undefined,
): string | null {
  const value = (input ?? "").trim();
  if (!value) {
    return null;
  }
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return null;
    }
    return parsed.hostname.toLowerCase() || null;
  } catch {
    return null;
  }
}

function isOpenAIManagedEndpoint(input: string | null | undefined): boolean {
  const host = parseHttpEndpointHost(input);
  return (
    host === "api.openai.com" ||
    host?.endsWith(".openai.azure.com") === true ||
    host?.endsWith(".services.ai.azure.com") === true
  );
}

export function resolveUiProviderTypeFromConfig(
  configProviderType: string,
  configDisplayName: string | null | undefined,
  configBaseUrl: string | null | undefined,
  _registryRows: ProviderRegistryEntry[],
  _existingProviderType: string | undefined,
): string {
  if (configProviderType !== CUSTOM_BACKEND_PROVIDER_TYPE) {
    return configProviderType;
  }
  if (isOpenAIManagedEndpoint(configBaseUrl)) {
    return CUSTOM_BACKEND_PROVIDER_TYPE;
  }
  const displayName = (configDisplayName ?? "").trim().toLowerCase();
  if (displayName === CUSTOM_PROVIDER_DISPLAY_NAME.toLowerCase()) {
    return LEGACY_CUSTOM_PROVIDER_TYPE;
  }
  const matchingCustomPreset = CUSTOM_PROVIDER_PRESETS.find(
    (preset) => preset.displayName.toLowerCase() === displayName,
  );
  if (matchingCustomPreset) {
    return matchingCustomPreset.providerType;
  }
  // A non-OpenAI host may proxy OpenAI, so without a legacy label we cannot infer custom.
  return configProviderType;
}

export function mergeLearnedModelCapabilities(
  stored: Record<string, { vision?: boolean; studio_tools?: boolean }> | undefined,
  registryCapabilities: Record<string, { vision?: boolean; studio_tools?: boolean }> | undefined,
  supportsStudioTools: boolean | undefined,
): Record<string, { vision?: boolean; studio_tools?: boolean }> {
  const fromRegistry = registryCapabilities ?? {};
  // Plan-listed slugs are learned at runtime, so the map cannot be rebuilt from the registry alone.
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
  // Anthropic has no entry: a -YYYYMMDD id is the canonical pre-4.6 name, not a snapshot.
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

/** Prunes the saved list before the emptiness test so all-retired selections fall through. */
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
    autoReloadModels:
      providerType === "llama_cpp" ? existing.autoReloadModels === true : undefined,
    newModelCheck: existing.newModelCheck,
    openaiContainerTtlMinutes:
      providerType === "openai" &&
      typeof existing.openaiContainerTtlMinutes === "number" &&
      existing.openaiContainerTtlMinutes >= 1
        ? Math.min(existing.openaiContainerTtlMinutes, 20)
        : synced.openaiContainerTtlMinutes,
  };
}



export async function syncExternalProvidersFromBackend(
  existingProviders: ExternalProviderConfig[],
  isCurrent?: () => boolean,
): Promise<ExternalProviderConfig[]> {
  const [registryRows, loadedConfigRows] = await Promise.all([
    listProviderRegistry(),
    listProviderConfigs(),
  ]);

  for (const entry of registryRows) {
    // Self-hosted ids are user-supplied, so studio_tools is parked under the wildcard.
    const capabilities = mergeLearnedModelCapabilities(
      getProviderModelCapabilities(entry.provider_type),
      entry.model_capabilities,
      entry.supports_studio_tools,
    );
    setProviderModelCapabilities(entry.provider_type, capabilities);
  }
  // Persisted capabilities outlive the backend, so prune providers the registry dropped.
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
      // Suggestions, not a listing.
      const defaultSet = new Set(defaultModels);
      if (
        serverAvailableModels.length === 0 ||
        (serverAvailableModels.length === defaultSet.size &&
          serverAvailableModels.every((id) => defaultSet.has(id)))
      ) {
        suggestedBaselines.add(config.id);
      } else {
        suggestedBaselines.delete(config.id);
      }
      if (!config.shared_by_owner && (needsModelBackfill || needsAvailableBackfill)) {
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
        // Only the stored type decides what the backend accepts (legacy rows differ from the UI type).
        backendProviderType: config.provider_type,
        name: config.display_name,
        baseUrl: config.base_url ?? "",
        ...connectionApiFields(config.api_type),
        models: resolvedModels,
        availableModels: resolvedAvailableModels,
        maxOutputTokens: config.max_output_tokens ?? undefined,
        extraHeaders: extraHeadersFromSaved(config.extra_headers),
        reasoningConfig:
          config.provider_type === "custom" &&
          config.api_type !== "responses" && config.api_type !== "systemone"
            ? normalizeCustomReasoningConfig(config.reasoning_config)
            : undefined,

        hasApiKey: config.has_api_key,
        shared: config.shared === true,
        sharedByOwner: config.shared_by_owner === true,

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
  void refreshProviderModelCatalogs(syncedProviders, isCurrent);
  void tickNewlyListedModels(
    syncedProviders,
    registryRows.find((entry) => entry.provider_type === "openai_codex"),
    isCurrent,
  );
  return syncedProviders;
}

/** Stored list is only the suggestions: adopt the next listing, tick nothing. */
export const suggestedBaselines = new Set<string>();

type NewlyListed = { models: string[]; availableModels: string[]; added: string[] };

/** null = no change; ids listed before and unticked since stay unticked. */
export function withNewlyListedModels(
  savedModels: string[],
  previouslyListed: string[],
  listedIds: string[],
  options: { knownIds?: string[]; maxAdded?: number } = {},
): NewlyListed | null {
  if (listedIds.length === 0 || savedModels.length === 0 || previouslyListed.length === 0) {
    return null;
  }
  const before = new Set([...previouslyListed, ...savedModels]);
  const added = listedIds.filter((id) => !before.has(id));
  if (added.length === 0) return null;
  const known = options.knownIds ? new Set(options.knownIds) : null;
  const stillKnown = (ids: string[]) => (known ? ids.filter((id) => known.has(id)) : ids);
  // Keep ids seen before, or one unticked while briefly unlisted would come back ticked.
  const seen = stillKnown(previouslyListed);
  if (options.maxAdded !== undefined && added.length > options.maxAdded) {
    // Baseline was the suggestions: adopt the listing, tick nothing.
    return {
      models: savedModels,
      availableModels: [...new Set([...listedIds, ...seen])],
      added: [],
    };
  }
  const models = [...stillKnown(savedModels), ...added];
  return { models, availableModels: [...new Set([...listedIds, ...models, ...seen])], added };
}

export function withNewlyOfferedCodexModels(
  savedModels: string[],
  previouslyOffered: string[],
  listed: CodexSubscriptionModels | null,
  maxAdded?: number,
): NewlyListed | null {
  if (listed?.source !== "subscription") return null;
  return withNewlyListedModels(
    savedModels,
    previouslyOffered,
    pruneProviderModelIds("openai_codex", listed.models.map((model) => model.id)),
    { knownIds: (listed.known ?? listed.models).map((model) => model.id), maxAdded },
  );
}

export function autoTickLimit(providerId: string, limit?: number): number | undefined {
  return suggestedBaselines.has(providerId) ? 0 : limit;
}

// Gateways list new third-party models every week; ticking each would flood the picker.
const NO_AUTO_TICK_PROVIDER_TYPES = new Set(["openrouter", "huggingface"]);
export const MAX_AUTO_TICKED_MODELS = 3;

export function autoTicksNewlyListedModels(
  provider: Pick<ExternalProviderConfig, "providerType" | "authKind" | "sharedByOwner">,
): boolean {
  return (
    provider.authKind !== "chatgpt_oauth" &&
    !provider.sharedByOwner &&
    !NO_AUTO_TICK_PROVIDER_TYPES.has(provider.providerType) &&
    !isCustomProviderType(provider.providerType) &&
    !supportsRemoteModelCatalog(provider.providerType)
  );
}

const NEW_MODEL_CHECK_PERIOD_MS: Partial<Record<NewModelCheckInterval, number>> = {
  daily: 24 * 60 * 60 * 1000,
  weekly: 7 * 24 * 60 * 60 * 1000,
};

/** Every focus re-syncs, so listings are rate-limited per connection. */
export function newModelCheckDue(
  interval: NewModelCheckInterval,
  checkedThisLaunch: boolean,
  lastCheckedAt: number | undefined,
  now: number,
): boolean {
  if (interval === "off") return false;
  const period = NEW_MODEL_CHECK_PERIOD_MS[interval];
  if (period === undefined) return !checkedThisLaunch;
  return lastCheckedAt === undefined || now - lastCheckedAt >= period;
}

const checkedThisLaunch = new Set<string>();
const ticksInFlight = new Set<string>();
// Failed or skipped checks retry after this, not on every focus.
const RETRY_AFTER_MS = 10 * 60 * 1000;
const lastAttemptAt = new Map<string, number>();

function markChecked(providerId: string): void {
  checkedThisLaunch.add(providerId);
  saveNewModelCheckAt(providerId, Date.now());
}

type NewlyListedFor = (row: ExternalProviderConfig) => NewlyListed | null;

async function listForNewModels(
  provider: ExternalProviderConfig,
  codexEntry: ProviderRegistryEntry | undefined,
): Promise<NewlyListedFor | null> {
  const isCodex = provider.authKind === "chatgpt_oauth";
  if (isCodex ? provider.authStatus !== "connected" : !autoTicksNewlyListedModels(provider)) {
    return null;
  }
  const now = Date.now();
  const due = newModelCheckDue(
    newModelCheckOf(currentRow(provider.id) ?? provider),
    checkedThisLaunch.has(provider.id),
    loadNewModelChecksAt()[provider.id],
    now,
  );
  const attemptedAt = lastAttemptAt.get(provider.id);
  if (!due || (attemptedAt !== undefined && now - attemptedAt < RETRY_AFTER_MS)) return null;
  lastAttemptAt.set(provider.id, now);
  if (isCodex) {
    const listed = await fetchCodexSubscriptionModels(provider.id);
    // A newly ticked text-only slug must not inherit the provider's vision default.
    const capabilities = codexCapabilitiesWithPlanModels(
      codexEntry,
      listed,
      getProviderModelCapabilities("openai_codex"),
    );
    if (capabilities) setProviderModelCapabilities("openai_codex", capabilities);
    if (listed.source === "subscription") {
      learnModelDisplayNames("openai_codex", listed.known ?? listed.models);
    }
    return (row) =>
      withNewlyOfferedCodexModels(
        row.models,
        row.availableModels ?? [],
        listed,
        autoTickLimit(row.id),
      );
  }
  const listed = await listProviderModels({
    providerType: toExternalBackendProviderType(provider.providerType),
    providerId: provider.id,
    apiKey: provider.hasApiKey ? "" : getExternalProviderApiKey(provider.id),
    baseUrl: provider.baseUrl || null,
    extraHeaders: provider.extraHeaders,
  });
  learnCatalogModelCapabilities(provider.providerType, listed);
  learnModelDisplayNames(provider.providerType, listed);
  const listedIds = pruneProviderModelIds(provider.providerType, [
    ...new Set(listed.map((model) => model.id.trim()).filter((id) => id.length > 0)),
  ]);
  return (row) =>
    withNewlyListedModels(row.models, row.availableModels ?? [], listedIds, {
      maxAdded: autoTickLimit(row.id, MAX_AUTO_TICKED_MODELS),
    });
}

export function codexCapabilitiesWithPlanModels(
  entry: ProviderRegistryEntry | undefined,
  listed: CodexSubscriptionModels | null,
  stored: Record<string, { vision?: boolean; studio_tools?: boolean }> | undefined,
): Record<string, { vision?: boolean; studio_tools?: boolean }> | null {
  if (!listed || listed.source !== "subscription") return null;
  // Without the registry row every model reads as unlisted and persisted capabilities go wrong.
  if (!entry) return null;
  const registryCapabilities = entry.model_capabilities ?? {};
  // Keyed by provider type, so merge into what is learned or a second connection drops the first's.
  const capabilities = mergeLearnedModelCapabilities(
    stored,
    registryCapabilities,
    entry?.supports_studio_tools,
  );
  for (const model of listed.known ?? listed.models) {
    // Only the plan describes unlisted slugs; mirror the backend bool(vision) gate as false, not null.
    if (!(model.id in registryCapabilities)) {
      capabilities[model.id] = {
        ...capabilities[model.id],
        vision: model.vision === true,
      };
    }
  }
  return capabilities;
}

const currentRow = (id: string) =>
  useExternalProvidersStore.getState().providers.find((candidate) => candidate.id === id);

async function tickNewlyListedModels(
  providers: readonly ExternalProviderConfig[],
  codexEntry: ProviderRegistryEntry | undefined,
  isCurrent?: () => boolean,
): Promise<void> {
  for (const provider of providers) {
    if (
      provider.sharedByOwner ||
      ticksInFlight.has(provider.id) ||
      providerSavesInFlight.has(provider.id)
    ) {
      continue;
    }
    ticksInFlight.add(provider.id);
    try {
      const updateFor = await listForNewModels(provider, codexEntry);
      if (isCurrent && !isCurrent()) return;
      if (!updateFor || providerSavesInFlight.has(provider.id)) continue;
      // A save may have landed during the listing.
      const latest = currentRow(provider.id);
      if (newModelCheckOf(latest ?? provider) === "off") continue;
      const update = updateFor(latest ?? provider);
      if (!update) {
        markChecked(provider.id);
        continue;
      }
      const { models, availableModels } = update;
      await updateProviderConfig(provider.id, { models, availableModels });
      markChecked(provider.id);
      if (isCurrent && !isCurrent()) return;
      if (currentRow(provider.id) !== latest) continue;
      const store = useExternalProvidersStore.getState();
      store.setProviders(
        store.providers.map((candidate) =>
          candidate.id === provider.id
            ? { ...candidate, models, availableModels }
            : candidate,
        ),
      );
    } catch {
      // Retried on a later sync.
    } finally {
      ticksInFlight.delete(provider.id);
    }
  }
}

const MODEL_CATALOG_TTL_MS = 24 * 60 * 60 * 1000;
// Only the provider's own endpoint may write the per-type live catalog, not a gateway.
const MODEL_CATALOG_PROVIDER_BASE_URLS: Record<string, string> = {
  openrouter: "https://openrouter.ai/api/v1",
};

function usesProviderCatalogEndpoint(provider: ExternalProviderConfig): boolean {
  const expected = MODEL_CATALOG_PROVIDER_BASE_URLS[provider.providerType];
  if (!expected) return false;
  const baseUrl = (provider.baseUrl ?? "").trim().replace(/\/+$/, "").toLowerCase();
  return baseUrl === "" || baseUrl === expected;
}

let modelsDevRefresh: Promise<ModelCatalogResponse> | null = null;

export async function refreshModelsDevCatalog(isCurrent?: () => boolean): Promise<boolean> {
  const modelsDevAge = modelsDevCatalogFetchedAt();
  if (modelsDevAge != null && Date.now() - modelsDevAge * 1000 < MODEL_CATALOG_TTL_MS) {
    return false;
  }
  try {
    modelsDevRefresh ??= fetchModelCatalog().finally(() => {
      modelsDevRefresh = null;
    });
    const catalog = await modelsDevRefresh;
    if (isCurrent && !isCurrent()) return false;
    setModelsDevCatalog(catalog);
    return true;
  } catch {
    // Offline: the bundled snapshot answers until the next sync.
    return false;
  }
}

export async function refreshProviderModelCatalogs(
  providers: readonly ExternalProviderConfig[],
  isCurrent?: () => boolean,
): Promise<void> {
  if (providers.length > 0) await refreshModelsDevCatalog(isCurrent);
  if (isCurrent && !isCurrent()) return;
  for (const provider of providers) {
    const providerType = provider.providerType;
    if (!usesProviderCatalogEndpoint(provider)) continue;
    const fetchedAt = providerModelCatalogFetchedAt(providerType);
    if (fetchedAt != null && Date.now() - fetchedAt < MODEL_CATALOG_TTL_MS) continue;
    try {
      const models = await listProviderModelCapabilities({
        providerType,
        providerId: provider.id,
        apiKey: "",
      });
      if (isCurrent && !isCurrent()) return;
      if (models.length > 0) setProviderModelCatalog(providerType, models);
    } catch {
      // Offline or unauthorized: the built-in tables answer until the next sync.
    }
  }
}
