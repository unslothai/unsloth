// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ConnectionApiType,
  ProviderApiType,
  ProviderAuthKind,
  ProviderAuthStatus,
} from "./api/providers-api";
import { modelCatalogSupportsVision } from "./model-catalog.ts";
import {
  type CustomReasoningConfig,
  normalizeCustomReasoningConfig,
} from "./custom-reasoning.ts";

export interface ExternalProviderConfig {
  id: string;
  providerType: string;
  name: string;
  baseUrl: string;
  apiType?: ProviderApiType;
  reasoningConfig?: CustomReasoningConfig;
  decisionsOnly?: boolean;
  models: string[];
  availableModels?: string[];
  /** The type as the backend stores it (may differ from providerType). Absent means unknown. */
  backendProviderType?: string;
  maxOutputTokens?: number;

  hasApiKey?: boolean;

  /** Sanitized backend-owned authorization state; never contains OAuth material. */
  authKind?: ProviderAuthKind;
  authStatus?: ProviderAuthStatus;
  enablePromptCaching?: boolean;
  /** Anthropic cache_control.ttl bucket; omitted inherits the 5-minute default. */
  promptCacheTtl?: "5m" | "1h";
  isReasoningModel?: boolean;
  autoReloadModels?: boolean;
  openaiContainerTtlMinutes?: number;
  createdAt: number;
  updatedAt: number;
}

// Gemini excluded: its caching needs a separate cachedContents POST not yet implemented.
const PROMPT_CACHING_PROVIDER_TYPES = new Set(["openai", "anthropic", "openrouter"]);

export function supportsProviderPromptCaching(
  providerType: string | null | undefined,
): boolean {
  return providerType != null && PROMPT_CACHING_PROVIDER_TYPES.has(providerType);
}

const PROMPT_CACHE_TTL_PROVIDER_TYPES = new Set(["anthropic", "openrouter"]);

export function supportsProviderPromptCacheTtl(
  providerType: string | null | undefined,
): boolean {
  return (
    providerType != null && PROMPT_CACHE_TTL_PROVIDER_TYPES.has(providerType)
  );
}

function cacheSettingsApplyToModel(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean {
  return providerType !== "openrouter" || /^~?anthropic\//i.test(modelId ?? "");
}

export function promptCachingAppliesToModel(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean {
  return (
    supportsProviderPromptCaching(providerType) &&
    cacheSettingsApplyToModel(providerType, modelId)
  );
}

export function promptCacheTtlAppliesToModel(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean {
  return (
    supportsProviderPromptCacheTtl(providerType) &&
    cacheSettingsApplyToModel(providerType, modelId)
  );
}

const PROMPT_CACHE_TTL_VALUES = new Set<"5m" | "1h">(["5m", "1h"]);

export function isPromptCacheTtl(value: unknown): value is "5m" | "1h" {
  return typeof value === "string" && PROMPT_CACHE_TTL_VALUES.has(value as "5m" | "1h");
}

// vLLM's OpenAI-compat endpoint does not advertise reasoning per model.
const REASONING_TOGGLE_PROVIDER_TYPES = new Set(["vllm"]);

export function supportsProviderReasoningToggle(
  providerType: string | null | undefined,
): boolean {
  return (
    providerType != null && REASONING_TOGGLE_PROVIDER_TYPES.has(providerType)
  );
}

const DECISION_PROVIDER_TYPES = new Set(["typesafe", "liquid"]);

export function isDecisionConnection(
  provider: Pick<ExternalProviderConfig, "providerType" | "decisionsOnly">,
): boolean {
  return (
    provider.decisionsOnly === true ||
    DECISION_PROVIDER_TYPES.has(provider.providerType)
  );
}

export function connectionApiFields(
  apiType: ConnectionApiType | undefined,
): Pick<ExternalProviderConfig, "apiType" | "decisionsOnly"> {
  return {
    apiType: apiType === "responses" ? "responses" : "chat_completions",
    decisionsOnly: apiType === "systemone",
  };
}

const NON_VISION_PROVIDER_TYPES = new Set<string>([
  "cohere",
  "deepseek",
  "mistral",
]);
const VISION_CAPABLE_PROVIDER_TYPES = new Set<string>([
  "openai",
  "anthropic",
  "gemini",
  "openrouter",
]);

// false = text-only, true = vision, null = unknown (default-allow).
export function providerTypeSupportsVision(
  providerType: string | null | undefined,
): boolean | null {
  if (providerType == null) return null;
  if (NON_VISION_PROVIDER_TYPES.has(providerType)) return false;
  if (VISION_CAPABLE_PROVIDER_TYPES.has(providerType)) return true;
  return null;
}


export type ProviderModelCapability = {
  vision?: boolean;
  studio_tools?: boolean;
  thinking?: boolean;
};

const REGISTRY_MODEL_CAPABILITIES = new Map<
  string,
  Record<string, ProviderModelCapability>
>();

const REGISTRY_MODEL_CAPABILITIES_KEY =
  "unsloth_chat_provider_model_capabilities";
let registryCapabilitiesHydrated = false;

function hydrateProviderModelCapabilities(): void {
  if (registryCapabilitiesHydrated) return;
  registryCapabilitiesHydrated = true;
  if (!canUseStorage()) return;
  try {
    const parsed = JSON.parse(
      localStorage.getItem(REGISTRY_MODEL_CAPABILITIES_KEY) ?? "{}",
    ) as Record<string, Record<string, ProviderModelCapability>>;
    for (const [providerType, capabilities] of Object.entries(parsed)) {
      if (capabilities && typeof capabilities === "object") {
        REGISTRY_MODEL_CAPABILITIES.set(providerType, capabilities);
      }
    }
  } catch {
    // Ignore invalid browser state; the backend registry will repopulate it.
  }
}

function persistProviderModelCapabilities(): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(
      REGISTRY_MODEL_CAPABILITIES_KEY,
      JSON.stringify(Object.fromEntries(REGISTRY_MODEL_CAPABILITIES)),
    );
  } catch {
    // Ignore storage failures; capabilities remain valid for this session.
  }
}

export function getProviderModelCapabilities(
  providerType: string,
): Record<string, ProviderModelCapability> | undefined {
  hydrateProviderModelCapabilities();
  return REGISTRY_MODEL_CAPABILITIES.get(providerType);
}

export function setProviderModelCapabilities(
  providerType: string,
  capabilities: Record<string, ProviderModelCapability> | undefined,
): void {
  hydrateProviderModelCapabilities();
  if (capabilities) REGISTRY_MODEL_CAPABILITIES.set(providerType, capabilities);
  else REGISTRY_MODEL_CAPABILITIES.delete(providerType);
  persistProviderModelCapabilities();
}

/** Drop capabilities for types the registry no longer lists, or stale flags latch forever. */
export function pruneProviderModelCapabilities(knownProviderTypes: Iterable<string>): void {
  hydrateProviderModelCapabilities();
  const known = new Set(knownProviderTypes);
  let removed = false;
  for (const providerType of [...REGISTRY_MODEL_CAPABILITIES.keys()]) {
    if (!known.has(providerType)) {
      REGISTRY_MODEL_CAPABILITIES.delete(providerType);
      removed = true;
    }
  }
  if (removed) persistProviderModelCapabilities();
}


// The backend strips image parts for these, so do not allow attachments.
const IMAGE_STRIPPING_PROVIDER_TYPES = new Set<string>(["deepseek"]);

export function providerModelSupportsVision(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean | null {

  hydrateProviderModelCapabilities();
  if (providerType && modelId) {
    const capability = REGISTRY_MODEL_CAPABILITIES.get(providerType)?.[modelId];
    if (typeof capability?.vision === "boolean") return capability.vision;
  }
  if (providerType != null && IMAGE_STRIPPING_PROVIDER_TYPES.has(providerType)) return false;
  const catalogVision = modelCatalogSupportsVision(providerType, modelId);
  if (catalogVision != null) return catalogVision;
  return providerTypeSupportsVision(providerType);
}


// Mirrors _MIXED_CATALOG_PROVIDER_TYPES in studio/backend/routes/inference.py.
const MIXED_CATALOG_PROVIDER_TYPES = new Set(["huggingface", "openrouter", "qwen"]);

/** Same rule as the backend's _external_takes_mcp_images, so stripped images are not resent. */
export function providerModelTakesMcpImages(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean {
  if (providerTypeSupportsVision(providerType) === false) return false;
  hydrateProviderModelCapabilities();
  if (providerType && modelId) {
    const capability = REGISTRY_MODEL_CAPABILITIES.get(providerType)?.[modelId];
    if (typeof capability?.vision === "boolean") return capability.vision;
  }
  if (providerType && MIXED_CATALOG_PROVIDER_TYPES.has(providerType)) return false;
  return true;
}

export const PROVIDER_CAPABILITY_WILDCARD = "*";

export function providerModelSupportsStudioTools(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean | null {
  if (!providerType) return null;
  hydrateProviderModelCapabilities();
  const capabilities = REGISTRY_MODEL_CAPABILITIES.get(providerType);
  if (modelId) {
    const value = capabilities?.[modelId]?.studio_tools;
    if (typeof value === "boolean") return value;
  }
  const providerDefault = capabilities?.[PROVIDER_CAPABILITY_WILDCARD]?.studio_tools;
  return typeof providerDefault === "boolean" ? providerDefault : null;
}

/** No wildcard: one Ollama host serves both kinds. `null` = never described, not a yes. */
export function providerModelSupportsThinking(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean | null {
  if (!providerType || !modelId) return null;
  hydrateProviderModelCapabilities();
  const value =
    REGISTRY_MODEL_CAPABILITIES.get(providerType)?.[modelId]?.thinking;
  return typeof value === "boolean" ? value : null;
}

export function learnCatalogModelCapabilities(
  providerType: string,
  models: readonly { id: string; capabilities?: string[] | null }[],
): void {
  if (!providerType) return;
  const stored = getProviderModelCapabilities(providerType) ?? {};
  const merged: Record<string, ProviderModelCapability> = { ...stored };
  let learned = false;
  for (const model of models) {
    const names = model.capabilities;
    const modelId = model.id?.trim();
    if (!modelId || !Array.isArray(names)) continue;
    merged[modelId] = {
      ...stored[modelId],
      thinking: names.includes("thinking"),
    };
    learned = true;
  }
  if (learned) setProviderModelCapabilities(providerType, merged);
}

export function externalModelSupportsStudioTools(
  checkpoint: string | null | undefined,
): boolean {
  const selection = parseExternalModelId(checkpoint);
  if (!selection) return false;
  const provider = loadExternalProviders().find(
    (candidate) => candidate.id === selection.providerId,
  );
  if (!provider) return false;
  return (
    providerModelSupportsStudioTools(provider.providerType, selection.modelId) === true
  );
}

export const CUSTOM_BACKEND_PROVIDER_TYPE = "openai";
export const LEGACY_CUSTOM_PROVIDER_TYPE = "custom";
export const CUSTOM_PROVIDER_DISPLAY_NAME = "Custom";
const OPENAI_CODEX_PROVIDER_TYPE = "openai_codex";
export const PROVIDER_MAX_OUTPUT_TOKENS_MIN = 64;

export function normalizeProviderMaxOutputTokens(
  value: unknown,
): number | undefined {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < PROVIDER_MAX_OUTPUT_TOKENS_MIN
  ) {
    return undefined;
  }
  return value;
}

/** All types except ChatGPT subscriptions; both stored and UI types are checked. */
export function supportsProviderMaxOutputTokens(
  uiProviderType: string | null | undefined,
  backendProviderType: string | null | undefined,
): boolean {
  if (!uiProviderType) return false;
  return (
    uiProviderType !== OPENAI_CODEX_PROVIDER_TYPE &&
    backendProviderType !== OPENAI_CODEX_PROVIDER_TYPE
  );
}

export const CUSTOM_PROVIDER_PRESETS = [
  {
    providerType: "llama_cpp",
    displayName: "llama.cpp",
    baseUrlPlaceholder: "http://localhost:8080/v1",
    modelIdsPlaceholder: "gpt-oss-20b\nqwen3-14b",
  },
  {
    providerType: "vllm",
    displayName: "vLLM",
    baseUrlPlaceholder: "https://my-vllm-server.com/v1",
    modelIdsPlaceholder: "openai/gpt-oss-20b\nQwen/Qwen3-14B",
  },
  {
    providerType: "ollama",
    displayName: "Ollama",
    baseUrlPlaceholder: "http://localhost:11434/v1",
    modelIdsPlaceholder: "gpt-oss:20b\nqwen3:14b",
  },
] as const;

const CUSTOM_PROVIDER_LABELS: Record<string, string> = {
  [LEGACY_CUSTOM_PROVIDER_TYPE]: CUSTOM_PROVIDER_DISPLAY_NAME,
  ...Object.fromEntries(
    CUSTOM_PROVIDER_PRESETS.map((preset) => [
      preset.providerType,
      preset.displayName,
    ]),
  ),
};

const CUSTOM_PROVIDER_BASE_URL_PLACEHOLDERS: Record<string, string> = {
  [LEGACY_CUSTOM_PROVIDER_TYPE]: "https://my-vllm-server.com/v1",
  ...Object.fromEntries(
    CUSTOM_PROVIDER_PRESETS.map((preset) => [
      preset.providerType,
      preset.baseUrlPlaceholder,
    ]),
  ),
};

const CUSTOM_PROVIDER_MODEL_IDS_PLACEHOLDERS: Record<string, string> = {
  [LEGACY_CUSTOM_PROVIDER_TYPE]: "openai/gpt-oss-20b\nQwen/Qwen3-14B",
  ...Object.fromEntries(
    CUSTOM_PROVIDER_PRESETS.map((preset) => [
      preset.providerType,
      preset.modelIdsPlaceholder,
    ]),
  ),
};

export function isCustomProviderType(
  providerType: string | null | undefined,
): boolean {
  if (!providerType) return false;
  return providerType in CUSTOM_PROVIDER_LABELS;
}

const REMOTE_MODEL_CATALOG_CUSTOM_PROVIDER_TYPES = new Set([
  LEGACY_CUSTOM_PROVIDER_TYPE,
  "ollama",
  "vllm",
  "llama_cpp",
]);

export function supportsRemoteModelCatalog(
  providerType: string | null | undefined,
): boolean {
  return (
    providerType != null &&
    REMOTE_MODEL_CATALOG_CUSTOM_PROVIDER_TYPES.has(providerType)
  );
}

/** Presets that hide the API-key field. Not Ollama: Ollama cloud requires a key. */
export function customPresetSkipsApiKeyField(
  providerType: string | null | undefined,
): boolean {
  return providerType === "llama_cpp";
}

export function allowsManualModelIdsWithCatalog(
  providerType: string | null | undefined,
): boolean {
  if (!providerType) return false;
  if (providerType === "openrouter") return true;
  return supportsRemoteModelCatalog(providerType);
}

export function customProviderDisplayName(
  providerType: string | null | undefined,
): string {
  if (!providerType) return CUSTOM_PROVIDER_DISPLAY_NAME;
  return CUSTOM_PROVIDER_LABELS[providerType] ?? providerType;
}

export function customProviderBaseUrlPlaceholder(
  providerType: string | null | undefined,
): string {
  if (!providerType) {
    return CUSTOM_PROVIDER_BASE_URL_PLACEHOLDERS[LEGACY_CUSTOM_PROVIDER_TYPE];
  }
  return (
    CUSTOM_PROVIDER_BASE_URL_PLACEHOLDERS[providerType] ??
    CUSTOM_PROVIDER_BASE_URL_PLACEHOLDERS[LEGACY_CUSTOM_PROVIDER_TYPE]
  );
}

export function customProviderModelIdsPlaceholder(
  providerType: string | null | undefined,
): string {
  if (!providerType) {
    return CUSTOM_PROVIDER_MODEL_IDS_PLACEHOLDERS[LEGACY_CUSTOM_PROVIDER_TYPE];
  }
  return (
    CUSTOM_PROVIDER_MODEL_IDS_PLACEHOLDERS[providerType] ??
    CUSTOM_PROVIDER_MODEL_IDS_PLACEHOLDERS[LEGACY_CUSTOM_PROVIDER_TYPE]
  );
}

export function toExternalBackendProviderType(providerType: string): string;
export function toExternalBackendProviderType(
  providerType: null | undefined,
): undefined;
export function toExternalBackendProviderType(
  providerType: string | null | undefined,
): string | undefined;
export function toExternalBackendProviderType(
  providerType: string | null | undefined,
): string | undefined {
  if (!providerType) return undefined;
  // vLLM /v1/responses 400s on strict-alternation templates; route it to chat completions.
  if (providerType === "vllm") return "vllm";
  if (providerType === "ollama") return "ollama";
  if (providerType === "llama_cpp") return "llama_cpp";
  if (providerType === LEGACY_CUSTOM_PROVIDER_TYPE) {
    return LEGACY_CUSTOM_PROVIDER_TYPE;
  }
  return isCustomProviderType(providerType)
    ? CUSTOM_BACKEND_PROVIDER_TYPE
    : providerType;
}

const EXTERNAL_PROVIDERS_KEY = "unsloth_chat_external_providers";
const EXTERNAL_PROVIDER_KEYS_KEY = "unsloth_chat_external_provider_keys";
const CONNECTIONS_ENABLED_KEY = "unsloth_chat_connections_enabled";
const EXTERNAL_MODEL_PREFIX = "external::";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

export function isExternalModelId(
  value: string | null | undefined,
): value is string {
  return typeof value === "string" && value.startsWith(EXTERNAL_MODEL_PREFIX);
}

export function buildExternalModelId(providerId: string, modelId: string): string {
  return `${EXTERNAL_MODEL_PREFIX}${providerId}::${encodeURIComponent(modelId)}`;
}

export function parseExternalModelId(
  value: string | null | undefined,
): { providerId: string; modelId: string } | null {
  if (!isExternalModelId(value)) return null;
  const payload = value.slice(EXTERNAL_MODEL_PREFIX.length);
  const separator = payload.indexOf("::");
  if (separator < 0) return null;
  const providerId = payload.slice(0, separator);
  const encodedModelId = payload.slice(separator + 2);
  if (!providerId || !encodedModelId) return null;
  try {
    return { providerId, modelId: decodeURIComponent(encodedModelId) };
  } catch {
    return null;
  }
}

function isExternalProviderConfig(value: unknown): value is ExternalProviderConfig {
  if (!value || typeof value !== "object") return false;
  const maybe = value as Partial<ExternalProviderConfig>;
  return (
    typeof maybe.id === "string" &&
    typeof maybe.providerType === "string" &&
    typeof maybe.name === "string" &&
    typeof maybe.baseUrl === "string" &&
    Array.isArray(maybe.models)
  );
}

function mapLegacyPresetToProviderType(presetId: string): string {
  if (presetId === "google") return "gemini";
  return presetId;
}

function normalizeProvider(raw: ExternalProviderConfig): ExternalProviderConfig {
  const providerType = raw.providerType.trim();
  return {
    ...raw,
    providerType,
    name: raw.name.trim(),
    baseUrl: raw.baseUrl.trim(),
    apiType: raw.apiType === "responses" ? "responses" : "chat_completions",
    reasoningConfig:
      providerType === "custom" &&
      (raw.backendProviderType === undefined || raw.backendProviderType === "custom") &&
      raw.apiType !== "responses" && raw.decisionsOnly !== true
        ? normalizeCustomReasoningConfig(raw.reasoningConfig)
        : undefined,
    models: raw.models
      .map((model) => model.trim())
      .filter((model) => model.length > 0),
    availableModels: (raw.availableModels ?? [])
      .map((model) => model.trim())
      .filter((model) => model.length > 0),
    backendProviderType:
      typeof raw.backendProviderType === "string" &&
      raw.backendProviderType.trim().length > 0
        ? raw.backendProviderType.trim()
        : undefined,
    maxOutputTokens: normalizeProviderMaxOutputTokens(raw.maxOutputTokens),
    enablePromptCaching: supportsProviderPromptCaching(providerType)
      ? raw.enablePromptCaching !== false
      : undefined,
    promptCacheTtl:
      supportsProviderPromptCacheTtl(providerType) &&
      isPromptCacheTtl(raw.promptCacheTtl)
        ? raw.promptCacheTtl
        : undefined,
    isReasoningModel: supportsProviderReasoningToggle(providerType)
      ? raw.isReasoningModel === true
      : undefined,
    autoReloadModels:
      providerType === "llama_cpp" ? raw.autoReloadModels === true : undefined,
    openaiContainerTtlMinutes:
      providerType === "openai" &&
      typeof raw.openaiContainerTtlMinutes === "number" &&
      raw.openaiContainerTtlMinutes >= 1
        ? Math.min(raw.openaiContainerTtlMinutes, 20)
        : undefined,
  };
}

function isCompleteProvider(provider: ExternalProviderConfig): boolean {
  if (!provider.id || !provider.name || !provider.providerType) return false;
  return true;
}

type LegacyProviderConfig = {
  id?: unknown;
  presetId?: unknown;
  name?: unknown;
  baseUrl?: unknown;
  models?: unknown;
  createdAt?: unknown;
  updatedAt?: unknown;
};

function fromUnknownProvider(value: unknown): ExternalProviderConfig | null {
  if (!value || typeof value !== "object") return null;
  if (isExternalProviderConfig(value)) {
    return value;
  }
  const legacy = value as LegacyProviderConfig;
  const id = typeof legacy.id === "string" ? legacy.id : "";
  const presetId = typeof legacy.presetId === "string" ? legacy.presetId : "";
  if (!id || !presetId || presetId === "custom") return null;
  const providerType = mapLegacyPresetToProviderType(presetId);
  if (!providerType) return null;
  return {
    id,
    providerType,
    name: typeof legacy.name === "string" ? legacy.name : providerType,
    baseUrl: typeof legacy.baseUrl === "string" ? legacy.baseUrl : "",
    models: Array.isArray(legacy.models)
      ? legacy.models.filter((item): item is string => typeof item === "string")
      : [],
    createdAt: typeof legacy.createdAt === "number" ? legacy.createdAt : Date.now(),
    updatedAt: typeof legacy.updatedAt === "number" ? legacy.updatedAt : Date.now(),
  };
}

export function loadConnectionsEnabled(): boolean {
  if (!canUseStorage()) return true;
  try {
    const raw = localStorage.getItem(CONNECTIONS_ENABLED_KEY);
    if (raw == null) return true;
    return raw === "true";
  } catch {
    return true;
  }
}

export function saveConnectionsEnabled(enabled: boolean): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(CONNECTIONS_ENABLED_KEY, enabled ? "true" : "false");
  } catch {
    // ignore
  }
}

export function loadExternalProviders(): ExternalProviderConfig[] {
  if (!canUseStorage()) return [];
  try {
    const raw = localStorage.getItem(EXTERNAL_PROVIDERS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map(fromUnknownProvider)
      .filter((provider): provider is ExternalProviderConfig => provider !== null)
      .map(normalizeProvider)
      .filter(isCompleteProvider);
  } catch {
    return [];
  }
}



function loadRawKeyMap(): Record<string, string> {
  if (!canUseStorage()) return {};
  try {
    const raw = localStorage.getItem(EXTERNAL_PROVIDER_KEYS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const out: Record<string, string> = {};
    for (const [providerId, value] of Object.entries(parsed)) {
      if (typeof providerId === "string" && typeof value === "string") {
        out[providerId] = value;
      }
    }
    return out;
  } catch {
    return {};
  }
}

function saveRawKeyMap(map: Record<string, string>): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(EXTERNAL_PROVIDER_KEYS_KEY, JSON.stringify(map));
  } catch {
    // ignore
  }
}

export function saveExternalProviders(
  providers: ExternalProviderConfig[],
): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(EXTERNAL_PROVIDERS_KEY, JSON.stringify(providers));
    // Keep unmatched legacy keys until the backend confirms the exact key was stored.
  } catch {
    // ignore
  }
}

export function getExternalProviderApiKey(
  providerId: string,
): string {

  const keys = loadRawKeyMap();
  return keys[providerId] ?? "";
}

export function pruneExternalProviderApiKeys(providerIds: Iterable<string>): void {
  if (!canUseStorage()) return;
  const retainedIds = new Set(providerIds);
  try {
    const keys = loadRawKeyMap();
    let changed = false;
    for (const providerId of Object.keys(keys)) {
      if (retainedIds.has(providerId)) continue;
      delete keys[providerId];
      changed = true;
    }
    if (changed) saveRawKeyMap(keys);
  } catch {
    // Keep legacy data untouched when storage is unavailable.
  }
}



export function removeExternalProviderApiKey(
  providerId: string,
  expectedApiKey?: string,
): void {
  if (!canUseStorage()) return;
  try {
    const keys = loadRawKeyMap();

    if (expectedApiKey !== undefined && keys[providerId] !== expectedApiKey) return;
    delete keys[providerId];
    saveRawKeyMap(keys);
  } catch {
    // ignore
  }
}
