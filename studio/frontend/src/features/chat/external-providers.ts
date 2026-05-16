// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export interface ExternalProviderConfig {
  id: string;
  /** Backend provider type (e.g. openai, mistral, gemini). */
  providerType: string;
  /** Display name in UI. */
  name: string;
  /** Provider base URL (default from registry or backend-saved override). */
  baseUrl: string;
  /** Model ids user enabled from `/api/providers/models`. */
  models: string[];
  /** Cached available model ids from the provider's /models response. */
  availableModels?: string[];
  /** Whether to ask supported hosted providers to use prompt caching. */
  enablePromptCaching?: boolean;
  /** User-pinned: the loaded vLLM model supports `enable_thinking`. */
  isReasoningModel?: boolean;
  /**
   * Default idle-timeout (in minutes) for newly created OpenAI shell
   * containers. Pre-fills the "Create container" dialog and is the
   * TTL the auto-create-per-thread path POSTs to /v1/containers with.
   * OpenAI's hard default is 20. Only meaningful for OpenAI cloud.
   */
  openaiContainerTtlMinutes?: number;
  createdAt: number;
  updatedAt: number;
}

const PROMPT_CACHING_PROVIDER_TYPES = new Set(["openai", "anthropic"]);

export function supportsProviderPromptCaching(
  providerType: string | null | undefined,
): boolean {
  return providerType != null && PROMPT_CACHING_PROVIDER_TYPES.has(providerType);
}

// Provider types that expose the connection-level "reasoning model"
// toggle. vLLM's OpenAI-compat endpoint doesn't advertise this per model.
const REASONING_TOGGLE_PROVIDER_TYPES = new Set(["vllm"]);

export function supportsProviderReasoningToggle(
  providerType: string | null | undefined,
): boolean {
  return (
    providerType != null && REASONING_TOGGLE_PROVIDER_TYPES.has(providerType)
  );
}

export const CUSTOM_BACKEND_PROVIDER_TYPE = "openai";
export const LEGACY_CUSTOM_PROVIDER_TYPE = "custom";

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
  [LEGACY_CUSTOM_PROVIDER_TYPE]: "Custom",
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

export function customProviderDisplayName(
  providerType: string | null | undefined,
): string {
  if (!providerType) return "Custom";
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
  // vLLM's /v1/responses applies the loaded model's chat template, which
  // 400s on strict-alternation templates (e.g. Gemma 3). Pass the actual
  // type through so the backend routes vLLM to /v1/chat/completions instead
  // of the OpenAI Responses path used for gpt-5.x.
  if (providerType === "vllm") return "vllm";
  return isCustomProviderType(providerType)
    ? CUSTOM_BACKEND_PROVIDER_TYPE
    : providerType;
}

const EXTERNAL_PROVIDERS_KEY = "unsloth_chat_external_providers";
const EXTERNAL_PROVIDER_KEYS_KEY = "unsloth_chat_external_provider_keys";
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
    models: raw.models
      .map((model) => model.trim())
      .filter((model) => model.length > 0),
    availableModels: (raw.availableModels ?? [])
      .map((model) => model.trim())
      .filter((model) => model.length > 0),
    enablePromptCaching: supportsProviderPromptCaching(providerType)
      ? raw.enablePromptCaching !== false
      : undefined,
    isReasoningModel: supportsProviderReasoningToggle(providerType)
      ? raw.isReasoningModel === true
      : undefined,
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

/**
 * Load the raw (encrypted or legacy plaintext) key map from localStorage.
 * Values are opaque strings — either AES-GCM ciphertext or legacy plaintext.
 */
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
    // Prune keys for removed providers — works on raw ciphertext, no decryption needed
    const allowedIds = new Set(providers.map((provider) => provider.id));
    const keys = loadRawKeyMap();
    const pruned: Record<string, string> = {};
    for (const [providerId, value] of Object.entries(keys)) {
      if (allowedIds.has(providerId)) {
        pruned[providerId] = value;
      }
    }
    saveRawKeyMap(pruned);
  } catch {
    // ignore
  }
}

/**
 * Retrieve a provider API key from localStorage.
 * Returns "" if no key is stored.
 */
export function getExternalProviderApiKey(
  providerId: string,
): string {
  const keys = loadRawKeyMap();
  return keys[providerId] ?? "";
}

/**
 * Store a provider API key in localStorage.
 */
export function setExternalProviderApiKey(
  providerId: string,
  apiKey: string,
): void {
  if (!canUseStorage()) return;
  const keys = loadRawKeyMap();
  keys[providerId] = apiKey;
  saveRawKeyMap(keys);
}

export function removeExternalProviderApiKey(providerId: string): void {
  if (!canUseStorage()) return;
  try {
    const keys = loadRawKeyMap();
    delete keys[providerId];
    saveRawKeyMap(keys);
  } catch {
    // ignore
  }
}
