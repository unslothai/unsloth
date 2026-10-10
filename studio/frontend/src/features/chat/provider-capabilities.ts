// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  normalizeProviderMaxOutputTokens,
  providerModelSupportsStudioTools,
  providerModelSupportsThinking,
} from "./external-providers";
import {
  type ModelCatalogEntry,
  REASONING_EFFORT_SCALE,
  type ReasoningEffortLevel,
  resolveModelCatalogEntry,
  resolveModelCatalogEntryByName,
  sortReasoningEfforts,
} from "./model-catalog";

import { normalizeCustomReasoningConfig } from "./custom-reasoning";

export { modelCatalogVersion, subscribeModelCatalog } from "./model-catalog";

/** Per-provider sampling matrix from provider docs (2026-05); null caps render everything. */

export interface ProviderCapabilities {
  /** Reasoning-class models (gpt-5.x / o3 via /v1/responses) reject it. */
  temperature: boolean;
  topP: boolean;
  topK: boolean;
  minP: boolean;
  repetitionPenalty: boolean;
  presencePenalty: boolean;
}

export type ExternalReasoningCapabilities = {
  supportsReasoning: boolean;
  reasoningStyle: "enable_thinking" | "reasoning_effort" | "enable_thinking_effort";
  reasoningAlwaysOn: boolean;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: readonly ReasoningEffortLevel[];
  defaultEffort?: ReasoningEffortLevel | null;
};

/** Clamp DOWNWARD to the nearest offered level; only below everything falls up.
 *  "none" is skipped unless asked for (matches llama_cpp.py). Legacy "xhigh" maps to "max". */
export function clampReasoningEffortToLevels(
  preferred: ExternalReasoningCapabilities["reasoningEffortLevels"][number],
  effortLevels: ExternalReasoningCapabilities["reasoningEffortLevels"],
): ExternalReasoningCapabilities["reasoningEffortLevels"][number] {
  let candidate = preferred;
  if (
    candidate === "xhigh" &&
    !effortLevels.includes("xhigh") &&
    effortLevels.includes("max")
  ) {
    candidate = "max";
  }
  if (effortLevels.includes(candidate)) {
    return candidate;
  }
  const rank = REASONING_EFFORT_SCALE.indexOf(
    candidate as (typeof REASONING_EFFORT_SCALE)[number],
  );
  if (rank !== -1) {
    const offered = REASONING_EFFORT_SCALE.filter((level) =>
      effortLevels.includes(level),
    );
    const rungs =
      candidate === "none" ? offered : offered.filter((level) => level !== "none");
    const searchable = rungs.length > 0 ? rungs : offered;
    const below = searchable.filter(
      (level) => REASONING_EFFORT_SCALE.indexOf(level) < rank,
    );
    const nearest = below.length > 0 ? below[below.length - 1] : searchable[0];
    if (nearest !== undefined) {
      return nearest;
    }
  }
  return effortLevels[0] ?? "low";
}

/** Only the reasoning_effort style sends a level; others send on/off. */
export function externalReasoningTakesEffort(
  caps: ExternalReasoningCapabilities,
): boolean {
  return caps.supportsReasoning && caps.reasoningStyle === "reasoning_effort";
}

/** Shared by switch, reload normalization and picker so they agree. `pinned: null` ignores pins. */
export function resolveExternalReasoningEffort(opts: {
  caps: ExternalReasoningCapabilities;
  providerType: string | null | undefined;
  apiType?: "chat_completions" | "responses";
  current: ReasoningEffortLevel;
  pinned?: string | null;
  restore?: boolean;
}): ReasoningEffortLevel {
  const { caps, current, pinned } = opts;
  const providerType = effectiveExternalReasoningProviderType(
    opts.providerType,
    opts.apiType,
  );
  const levels = caps.reasoningEffortLevels;
  // A style that sends no level must not move the chat's shared level.
  if (!externalReasoningTakesEffort(caps)) return current;
  if (pinned && levels.includes(pinned as ReasoningEffortLevel)) {
    return pinned as ReasoningEffortLevel;
  }
  if (opts.restore) return clampReasoningEffortToLevels(current, levels);
  if (caps.defaultEffort && levels.includes(caps.defaultEffort)) {
    return caps.defaultEffort;
  }
  const clamped = clampReasoningEffortToLevels(current, levels);
  // Anthropic: highest (adaptive thinking); OpenAI: high; others: medium.
  if (providerType === "anthropic") {
    return levels.includes("xhigh")
      ? "xhigh"
      : levels.includes("high")
        ? "high"
        : clamped;
  }
  if (providerType === "openai") {
    return levels.includes("high")
      ? "high"
      : levels.includes("medium")
        ? "medium"
        : clamped;
  }
  return levels.includes("medium") ? "medium" : clamped;
}

export const EXTERNAL_MAX_OUTPUT_TOKENS = 32768;

const EXTERNAL_MAX_OUTPUT_TOKENS_BY_MODEL: Array<{
  providerType: string;
  prefixes: readonly string[];
  cap: number;
}> = [
  // Responses API rejects over-limit max_output_tokens. First match wins: `-chat-latest` first,
  // bare family prefixes last.
  {
    providerType: "openai",
    prefixes: [
      "gpt-5.3-chat-latest",
      "gpt-5.2-chat-latest",
      "gpt-5.1-chat-latest",
      "gpt-5-chat-latest",
    ],
    cap: 16384,
  },
  {
    providerType: "openai",
    prefixes: ["gpt-5.6", "gpt-5.5-pro", "gpt-5.5", "gpt-5.2", "gpt-5.1"],
    cap: 128000,
  },
  { providerType: "openai", prefixes: ["gpt-5.4-pro", "gpt-5.4"], cap: 65536 },
  { providerType: "openai", prefixes: ["gpt-5.3"], cap: 16384 },
  { providerType: "openai", prefixes: ["gpt-5"], cap: 128000 },
  { providerType: "openai", prefixes: ["gpt-4.1"], cap: 32768 },
  { providerType: "openai", prefixes: ["gpt-4.5"], cap: 16384 },
  { providerType: "openai", prefixes: ["gpt-4o", "chatgpt-4o"], cap: 16384 },
  // Under the 8,192 default, so these two fail on an untouched config.
  {
    providerType: "openai",
    prefixes: ["gpt-3.5-turbo", "gpt-4-turbo"],
    cap: 4096,
  },
  { providerType: "openai", prefixes: ["gpt-4"], cap: 8192 },
  {
    providerType: "anthropic",
    prefixes: [
      "claude-opus-5",
      "claude-sonnet-5",
      "claude-fable-5",
      "claude-mythos-5",
      "claude-opus-4-8",
      "claude-opus-4-7",
    ],
    cap: 128000,
  },
  {
    providerType: "anthropic",
    prefixes: [
      "claude-opus-4-6",
      "claude-sonnet-4-6",
      "claude-opus-4-5",
      "claude-sonnet-4-5",
      "claude-haiku-4-5",
      "claude-sonnet-4-20250514",
    ],
    cap: 64000,
  },
  {
    providerType: "anthropic",
    prefixes: ["claude-opus-4-1", "claude-opus-4-20250514"],
    cap: 32000,
  },
  {
    providerType: "gemini",
    prefixes: ["gemini-3", "gemini-pro", "gemini-flash"],
    cap: 65536,
  },
  { providerType: "deepseek", prefixes: ["deepseek"], cap: 384000 },
];

/** A documented cap bounds the override rather than replacing it; output floor wins. */
export function getExternalMaxOutputTokens(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
  connectionMaxOutputTokens?: number | null,
): number {
  const override = normalizeProviderMaxOutputTokens(connectionMaxOutputTokens);
  const documented = _documentedMaxOutputTokens(providerType, modelId);
  const resolved =
    documented != null
      ? Math.min(documented, override ?? documented)
      : (override ?? EXTERNAL_MAX_OUTPUT_TOKENS);
  return Math.max(resolved, getExternalMinOutputTokens(providerType));
}

/**
 * Null unless a published cap or user override grounds it; guesses are unsafe as a budget.
 */
export function getGroundedExternalMaxOutputTokens(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
  connectionMaxOutputTokens?: number | null,
): number | null {
  const override = normalizeProviderMaxOutputTokens(connectionMaxOutputTokens);
  if (override == null && _publishedMaxOutputTokens(providerType, modelId) == null) return null;
  return getExternalMaxOutputTokens(providerType, modelId, connectionMaxOutputTokens);
}

export function externalMaxOutputTokensNeedsConnectionCap(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean {
  return _publishedMaxOutputTokens(providerType, modelId) == null;
}

/**
 * Published limit before the override, so a capped model is not mistaken for a small one.
 */
export function getPublishedExternalMaxOutputTokens(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): number | null {
  return _publishedMaxOutputTokens(providerType, modelId);
}

/**
 * Skips OpenRouter ids, which resolve via the direct provider's table but serve lower caps.
 */
function _publishedMaxOutputTokens(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): number | null {
  if (providerType === "openrouter") {
    return resolveModelCatalogEntry(providerType, modelId)?.maxOutputTokens ?? null;
  }
  return _documentedMaxOutputTokens(providerType, modelId);
}

function _documentedMaxOutputTokens(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): number | null {
  if (!providerType || !modelId) return null;
  const normalized = modelId.trim().toLowerCase();
  if (!normalized) return null;
  if (providerType === "openrouter") {
    const live = resolveModelCatalogEntry(providerType, normalized)?.maxOutputTokens;
    if (live != null) return live;
  }
  const stripped =
    providerType === "openrouter" && normalized.includes("/")
      ? normalized.split("/").slice(-1)[0]
      : normalized;
  const effectiveProvider =
    providerType === "openrouter"
      ? _inferProviderFromOpenrouterId(normalized) ?? providerType
      : providerType === "openai_codex"
        ? "openai_codex"
        : providerType;
  if (effectiveProvider === "openai_codex") return 128000;

  for (const entry of EXTERNAL_MAX_OUTPUT_TOKENS_BY_MODEL) {
    if (entry.providerType !== effectiveProvider) continue;
    if (entry.prefixes.some((prefix) => stripped.startsWith(prefix))) {
      return entry.cap;
    }
  }
  return null;
}

/** Only lowers and is persisted; no provider means unknown, not the 32,768 fallback. */
export function resolveExternalMaxTokensClamp(input: {
  settingsHydrated: boolean;
  hasActiveExternalProvider: boolean;
  isExternalModel: boolean;
  maxTokens: number;
  maxTokensMax: number;
}): number | null {
  if (!input.settingsHydrated || !input.hasActiveExternalProvider) return null;
  if (!input.isExternalModel || input.maxTokens <= input.maxTokensMax) {
    return null;
  }
  return input.maxTokensMax;
}

function _inferProviderFromOpenrouterId(
  normalizedId: string,
): string | null {
  if (normalizedId.startsWith("openai/")) return "openai";
  if (normalizedId.startsWith("anthropic/")) return "anthropic";
  if (normalizedId.startsWith("google/")) return "gemini";
  if (normalizedId.startsWith("deepseek/")) return "deepseek";
  return null;
}

/** Mistral excluded: its connector is Agents-API only and errors on /v1/chat/completions. */
export function providerSupportsBuiltinWebSearch(
  providerType: string | null | undefined,
  modelId?: string | null | undefined,
  baseUrl?: string | null | undefined,
): boolean {
  // Gemini 3 image models support Search grounding; older image ids and compat proxies do not.
  if (providerType === "gemini") {
    if (isGeminiCustomOpenAICompatBase(baseUrl)) return false;
    const normalized = modelId?.trim().toLowerCase() ?? "";
    if (normalized && isGeminiImageModel(normalized)) {
      return geminiImageModelAllowsGoogleSearch(normalized);
    }
    return true;
  }
  if (providerType === "openai_codex") {
    return providerModelSupportsStudioTools(providerType, modelId) === true;
  }

  return (
    providerType === "openai" ||
    providerType === "anthropic" ||
    providerType === "openrouter" ||
    providerType === "kimi"
  );
}

export function providerSupportsBuiltinWebFetch(
  providerType: string | null | undefined,
): boolean {
  return providerType === "anthropic";
}

/** Opus 5 / 4.8 only: 4.7 errors on `speed`, 4.6 accepts it but runs at standard speed. */
const ANTHROPIC_FAST_MODE_MODEL_PREFIXES = [
  "claude-opus-5",
  "claude-opus-4-8",
] as const;

export function providerSupportsFastMode(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean {
  if (providerType !== "anthropic") return false;
  if (!modelId) return false;
  // Family boundary required so ids like "claude-opus-4-70" do not match.
  return ANTHROPIC_FAST_MODE_MODEL_PREFIXES.some(
    (prefix) => modelId === prefix || modelId.startsWith(`${prefix}-`),
  );
}

const ANTHROPIC_CODE_EXECUTION_MODEL_PREFIXES = [
  "claude-opus-5",
  "claude-sonnet-5",
  "claude-fable-5",
  "claude-mythos-5",
  "claude-opus-4-8",
  "claude-opus-4-7",
  "claude-opus-4-6",
  "claude-sonnet-4-6",
  "claude-opus-4-5",
  "claude-sonnet-4-5",
  "claude-haiku-4-5",
  // Deprecated upstream but still in the registry.
  "claude-opus-4-1",
  "claude-opus-4",
  "claude-sonnet-4",
] as const;

// Check `gpt-5.5-pro` first so the prefix cannot collide with other 5.5 ids.
const OPENAI_CODE_EXECUTION_MODEL_PREFIXES = [
  "gpt-5.6",
  "gpt-5.5-pro",
  "gpt-5.5",
] as const;

/** Shell and image tools 400 on custom compat backends. Mirrors _is_openai_family_cloud. */
function isAzureOpenAICloudHost(host: string): boolean {
  return (
    host.endsWith(".openai.azure.com") ||
    host.endsWith(".services.ai.azure.com")
  );
}

function isOpenAICloudBaseUrl(baseUrl: string | null | undefined): boolean {
  if (!baseUrl) return true;
  try {
    const host = new URL(baseUrl).hostname.toLowerCase();
    return host === "api.openai.com" || isAzureOpenAICloudHost(host);
  } catch {
    return false;
  }
}

function isAzureOpenAICloudBaseUrl(baseUrl: string | null | undefined): boolean {
  if (!baseUrl) return false;
  try {
    return isAzureOpenAICloudHost(new URL(baseUrl).hostname.toLowerCase());
  } catch {
    return false;
  }
}

function usesOpenAIHostedResponses(
  providerType: string | null | undefined,
  baseUrl: string | null | undefined,
  apiType: "chat_completions" | "responses" | undefined,
): boolean {
  if (providerType === "openai") {
    return isOpenAICloudBaseUrl(baseUrl);
  }
  return (
    providerType === "custom" &&
    apiType === "responses" &&
    Boolean(baseUrl?.trim()) &&
    isOpenAICloudBaseUrl(baseUrl)
  );
}

export function providerSupportsBuiltinCodeExecution(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
  baseUrl?: string | null,
  apiType?: "chat_completions" | "responses",
): boolean {
  const normalized = modelId?.trim().toLowerCase() ?? "";
  if (!normalized) return false;
  if (providerType === "anthropic") {
    return ANTHROPIC_CODE_EXECUTION_MODEL_PREFIXES.some((prefix) =>
      normalized.startsWith(prefix),
    );
  }
  if (usesOpenAIHostedResponses(providerType, baseUrl, apiType)) {
    return OPENAI_CODE_EXECUTION_MODEL_PREFIXES.some((prefix) =>
      normalized.startsWith(prefix),
    );
  }
  if (providerType === "gemini") {
    // Gemini image ids reject text tools and compat proxies skip the native translator.
    if (isGeminiCustomOpenAICompatBase(baseUrl)) return false;
    if (isGeminiImageModel(normalized)) return false;
    return normalized.startsWith("gemini-");
  }
  return false;
}

const PROVIDER_TYPES_WITH_CODE_SANDBOX = new Set(["openai", "anthropic", "gemini"]);

export function providerHostsCodeExecution(
  providerType: string | null | undefined,
  baseUrl?: string | null,
  apiType?: "chat_completions" | "responses",
): boolean {
  return (
    PROVIDER_TYPES_WITH_CODE_SANDBOX.has(providerType ?? "") ||
    (providerType === "custom" &&
      usesOpenAIHostedResponses(providerType, baseUrl, apiType))
  );
}

const OPENAI_IMAGE_GENERATION_MODEL_PREFIXES = [
  "gpt-5.5-pro",
  "gpt-5.5",
  "gpt-5.4-pro",
  "gpt-5.4",
  "gpt-5.3",
  "gpt-5.2",
  "gpt-5.1",
  "gpt-5",
  "o3",
] as const;

export function providerSupportsBuiltinImageGeneration(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
  baseUrl?: string | null,
  apiType?: "chat_completions" | "responses",
): boolean {
  const normalized = modelId?.trim().toLowerCase() ?? "";
  if (!normalized) return false;
  if (usesOpenAIHostedResponses(providerType, baseUrl, apiType)) {
    return OPENAI_IMAGE_GENERATION_MODEL_PREFIXES.some((prefix) =>
      normalized.startsWith(prefix),
    );
  }
  if (providerType === "gemini") {
    // Compat proxies skip the native translator, so hide the pill.
    if (isGeminiCustomOpenAICompatBase(baseUrl)) return false;
    return normalized.includes("-image") || normalized.includes("nano-banana");
  }
  return false;
}

/** Mirrors the backend's is_image_picker_model. */
function isGeminiImageModel(modelId: string): boolean {
  const m = modelId.toLowerCase();
  return m.includes("-image") || m.includes("nano-banana");
}

export function isGeminiCustomOpenAICompatBase(
  baseUrl: string | null | undefined,
): boolean {
  if (!baseUrl) return false;
  try {
    const host = new URL(baseUrl).hostname.toLowerCase();
    return host.length > 0 && host !== "generativelanguage.googleapis.com";
  } catch {
    return false;
  }
}

/** Native Gemini rejects oversized input up front, so length stops mean Max Tokens. */
export function externalStopWindow(
  providerType: string | null | undefined,
  baseUrl: string | null | undefined,
): number | null {
  return providerType === "gemini" && !isGeminiCustomOpenAICompatBase(baseUrl)
    ? Number.POSITIVE_INFINITY
    : null;
}

/** Documented on Gemini 3 image models; older ids reject Search as a tool. */
function geminiImageModelAllowsGoogleSearch(modelId: string): boolean {
  const m = modelId.toLowerCase();
  return (
    m.startsWith("gemini-3-pro-image") ||
    m.startsWith("gemini-3.1-flash-image") ||
    m.startsWith("nano-banana-pro") ||
    m.startsWith("nano-banana-2")
  );
}

/** Kimi needs >= 16000 on thinking models so reasoning and answer both fit. */
const EXTERNAL_MIN_OUTPUT_TOKENS_BY_PROVIDER: Record<string, number> = {
  kimi: 16000,
};

export function getExternalMinOutputTokens(
  providerType: string | null | undefined,
): number {
  if (!providerType) return 64;
  return EXTERNAL_MIN_OUTPUT_TOKENS_BY_PROVIDER[providerType] ?? 64;
}

const OPENAI_COMPAT_BASE: ProviderCapabilities = {
  temperature: true,
  topP: true,
  topK: false,
  minP: false,
  repetitionPenalty: false,
  presencePenalty: true,
};

const CUSTOM_RESPONSES_CAPABILITIES: ProviderCapabilities = {
  ...OPENAI_COMPAT_BASE,
  // The Responses translator drops presence_penalty.
  presencePenalty: false,
};

const ALL_SUPPORTED: ProviderCapabilities = {
  temperature: true,
  topP: true,
  topK: true,
  minP: true,
  repetitionPenalty: true,
  presencePenalty: true,
};

const PROVIDER_CAPABILITIES: Record<string, ProviderCapabilities> = {
  // Reasoning-class ids on /v1/responses reject temperature, top_p and penalties.
  openai_codex: {
    temperature: false,
    topP: false,
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
  },

  openai: {
    temperature: false,
    topP: false,
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
  },
  anthropic: {
    temperature: true,
    topP: false,
    topK: true,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
  },
  mistral: OPENAI_COMPAT_BASE,
  // See https://ai.google.dev/api/rest/v1beta/GenerationConfig.
  gemini: {
    temperature: true,
    topP: true,
    topK: true,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: true,
  },
  // Kimi k2.5/k2.6 400 on non-default temperature/top_p; the backend also strips them.
  kimi: {
    temperature: false,
    topP: false,
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: true,
  },
  deepseek: {
    temperature: true,
    topP: true,
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
  },
  qwen: OPENAI_COMPAT_BASE,
  huggingface: OPENAI_COMPAT_BASE,
  // OpenRouter silently drops unsupported params, so surface every knob.
  openrouter: ALL_SUPPORTED,
  // OpenAI-shaped gateways 400 on unrecognized fields.
  custom: OPENAI_COMPAT_BASE,
  vllm: ALL_SUPPORTED,
  // Ollama /v1 silently drops top_k / min_p / repeat_penalty.
  ollama: OPENAI_COMPAT_BASE,
  llama_cpp: ALL_SUPPORTED,
};

const DEFAULT_EXTERNAL_CAPABILITIES = OPENAI_COMPAT_BASE;

// Mirrors _anthropic_sampling_params_removed in external_provider.py; a backend test checks it.
const ANTHROPIC_SAMPLING_REMOVED_MODEL =
  /^claude-(?:mythos-preview(?:-|$)|[a-z]+-(?:[5-9]|\d{2,})(?:[-.]|$)|opus-4[-.](?:0?[7-9]|[1-9]\d)(?:[-.]|$))/;

const OPENAI_RESPONSES_FIXED_SAMPLING_MODEL =
  /^(?:gpt-5(?:[.-]|$)|gpt-4\.5(?:[.-]|$)|o\d+(?:[.-]|$)|codex-mini(?:[.-]|$)|gpt-6-astra(?:[.-]|$))/;

export function getProviderCapabilities(
  providerType: string | null | undefined,
  apiType?: "chat_completions" | "responses",
  modelId?: string | null,
  baseUrl?: string | null,
): ProviderCapabilities | null {
  if (!providerType) return null;
  if (providerType === "custom" && apiType === "responses") {
    // Azure deployment names may hide the model, so suppress sampling there.
    const model = modelId?.trim().toLowerCase() ?? "";
    if (
      usesOpenAIHostedResponses(providerType, baseUrl, apiType) &&
      (isAzureOpenAICloudBaseUrl(baseUrl) ||
        (!OPENAI_NON_REASONING_CHAT_ALIAS.test(model) &&
          OPENAI_RESPONSES_FIXED_SAMPLING_MODEL.test(model)))
    ) {
      return PROVIDER_CAPABILITIES.openai;
    }
    return CUSTOM_RESPONSES_CAPABILITIES;
  }
  if (
    providerType === "anthropic" &&
    ANTHROPIC_SAMPLING_REMOVED_MODEL.test(modelId?.trim().toLowerCase() ?? "")
  ) {
    return {
      ...PROVIDER_CAPABILITIES.anthropic,
      temperature: false,
      topK: false,
    };
  }
  return PROVIDER_CAPABILITIES[providerType] ?? DEFAULT_EXTERNAL_CAPABILITIES;
}

const DEFAULT_EFFORT_LEVELS = ["low", "medium", "high"] as const;
const OPENROUTER_MANDATORY_REASONING_MODELS = new Set([
  "google/gemini-pro-latest",
  "baidu/cobuddy:free",
  "inclusionai/ring-2.6-1t:free",
  "deepseek/deepseek-r1",
]);

function isOpenRouterMandatoryReasoningModel(modelId: string): boolean {
  const normalized = modelId.trim().toLowerCase();
  const canonical = normalized.startsWith("~") ? normalized.slice(1) : normalized;
  return OPENROUTER_MANDATORY_REASONING_MODELS.has(canonical);
}
type ReasoningCaps = {
  supportsReasoning: boolean;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: ExternalReasoningCapabilities["reasoningEffortLevels"];
};

const DEFAULT_EXTERNAL_REASONING_CAPABILITIES: ExternalReasoningCapabilities = {
  supportsReasoning: false,
  reasoningStyle: "enable_thinking",
  reasoningAlwaysOn: false,
  supportsReasoningOff: false,
  reasoningEffortLevels: DEFAULT_EFFORT_LEVELS,
};

const NO_REASONING_CAPS: ReasoningCaps = {
  supportsReasoning: false,
  supportsReasoningOff: false,
  reasoningEffortLevels: DEFAULT_EFFORT_LEVELS,
};

const ANTHROPIC_REASONING_MODELS = [
  {
    // Fable, Mythos 5, and Opus 5.5 return 400 for disabled thinking; no off switch.
    prefixes: ["claude-fable-5", "claude-mythos-5", "claude-opus-5-5"],
    supportsOff: false,
    levels: ["low", "medium", "high", "xhigh", "max"],
  },
  {
    prefixes: [
      "claude-opus-5",
      "claude-sonnet-5",
      "claude-opus-4-8",
      "claude-opus-4-7",
    ],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "xhigh", "max"],
  },
  {
    prefixes: ["claude-opus-4-6", "claude-sonnet-4-6"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "max"],
  },
  {
    prefixes: ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high"],
  },
  {
    // Must come after the 4-5..4-8 entries so those match first.
    prefixes: [
      "claude-opus-4-1",
      "claude-opus-4-0",
      "claude-opus-4-2025",
      "claude-sonnet-4-0",
      "claude-sonnet-4-2025",
      "claude-3-7-sonnet",
    ],
    supportsOff: true,
    levels: ["none", "low", "medium", "high"],
  },
] as const;

function matchesModelPrefix(
  modelId: string,
  prefixes: readonly string[],
): boolean {
  // Mirrors `_anthropic_spec_prefix_matches` in external_provider.py.
  return prefixes.some((prefix) => {
    if (modelId === prefix) return true;
    if (!modelId.startsWith(prefix)) return false;
    const rest = modelId.slice(prefix.length);
    if (rest.startsWith("-")) return true;
    const trailingDigits = /\d+$/.exec(prefix)?.[0] ?? "";
    return trailingDigits.length >= 4 && /^\d/.test(rest);
  });
}

function resolveAnthropicReasoningEffortCapabilities(modelId: string): ReasoningCaps {
  const normalized = modelId.trim().toLowerCase();
  const matched = ANTHROPIC_REASONING_MODELS.find((entry) =>
    matchesModelPrefix(normalized, entry.prefixes),
  );
  if (matched) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: matched.supportsOff,
      reasoningEffortLevels: matched.levels,
    };
  }
  return NO_REASONING_CAPS;
}

const OPENAI_REASONING_MODELS = [
  {
    prefixes: ["gpt-6-astra"],
    supportsOff: false,
    levels: ["low", "medium", "high", "xhigh", "max"],
  },
  {
    prefixes: ["gpt-5.5-pro", "gpt-5.4-pro"],
    supportsOff: false,
    levels: ["medium", "high", "xhigh"],
  },
  {
    // Rejects "minimal".
    prefixes: ["gpt-5.6", "gpt-5.5", "gpt-5.4"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "xhigh"],
  },
  {
    prefixes: ["gpt-5.3-codex"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "xhigh"],
  },
  // 5.1+ 400s on "minimal"; must sit ahead of bare `gpt-5`.
  {
    prefixes: ["gpt-5.2"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "xhigh"],
  },
  // 5.1 Codex 400s on `none`; codex-max sorts first or plain codex swallows it.
  {
    prefixes: ["gpt-5.1-codex-max"],
    supportsOff: false,
    levels: ["low", "medium", "high", "xhigh"],
  },
  {
    prefixes: ["gpt-5.1-codex"],
    supportsOff: false,
    levels: ["low", "medium", "high"],
  },
  {
    prefixes: ["gpt-5.1"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high"],
  },
  {
    prefixes: ["gpt-5-codex"],
    supportsOff: false,
    levels: ["low", "medium", "high"],
  },
  {
    prefixes: ["gpt-5"],
    supportsOff: false,
    levels: ["minimal", "low", "medium", "high"],
  },
  {
    prefixes: ["o3"],
    supportsOff: false,
    levels: DEFAULT_EFFORT_LEVELS,
  },
] as const;

/** Non-reasoning chat aliases reject reasoning.effort; matched by suffix since prefixes swallow them. */
const OPENAI_NON_REASONING_CHAT_ALIAS = /-chat(?:-latest)?$/;

function resolveOpenAIReasoningEffortCapabilities(modelId: string): ReasoningCaps {
  const normalized = modelId.trim().toLowerCase();
  if (OPENAI_NON_REASONING_CHAT_ALIAS.test(normalized)) return NO_REASONING_CAPS;
  const matched = OPENAI_REASONING_MODELS.find((entry) =>
    matchesModelPrefix(normalized, entry.prefixes),
  );
  if (matched) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: matched.supportsOff,
      reasoningEffortLevels: matched.levels,
    };
  }
  return NO_REASONING_CAPS;
}

function withEnableThinkingStyle(
  overrides?: Partial<ExternalReasoningCapabilities>,
): ExternalReasoningCapabilities {
  return {
    ...DEFAULT_EXTERNAL_REASONING_CAPABILITIES,
    ...overrides,
    reasoningStyle: "enable_thinking",
  };
}

function withReasoningEffortStyle(caps: ReasoningCaps): ExternalReasoningCapabilities {
  return {
    ...DEFAULT_EXTERNAL_REASONING_CAPABILITIES,
    supportsReasoning: true,
    reasoningStyle: "reasoning_effort",
    supportsReasoningOff: caps.supportsReasoningOff,
    reasoningEffortLevels: caps.reasoningEffortLevels,
  };
}

function resolveKimiReasoningCapabilities(modelId: string): ExternalReasoningCapabilities {
  // Kimi has a boolean toggle: k2.6 toggleable, k2-thinking always on, others none.
  if (modelId === "kimi-k2-thinking") {
    return withEnableThinkingStyle({
      supportsReasoning: true,
      reasoningAlwaysOn: true,
    });
  }
  if (modelId === "kimi-k2.6") {
    return withEnableThinkingStyle({
      supportsReasoning: true,
      supportsReasoningOff: true,
    });
  }
  return withEnableThinkingStyle();
}

// Gemini 3.x uses thinkingLevel, 2.5 uses thinkingBudget. Mirrors _GEMINI3_FAMILY/_GEMINI3_PRO.
const GEMINI3_PRO_PATTERN = /^gemini-3(\.\d+)?-pro/;
const GEMINI3_FLASH_PATTERN = /^gemini-3(\.\d+)?-flash/;
const GEMINI3_PRO_PREFIXES = ["gemini-pro-latest"];
const GEMINI3_FLASH_PREFIXES = [
  "gemini-flash-latest",
  "gemini-flash-lite-latest",
];
const GEMINI25_PRO_PREFIXES = [
  "gemini-2.5-pro",
];
const GEMINI25_FLASH_PREFIXES = [
  "gemini-2.5-flash",
];
const GEMINI_IMAGE_HINTS = [
  "-image",
  "nano-banana",
];
function resolveGeminiReasoningCapabilities(
  modelId: string,
): ExternalReasoningCapabilities {
  const m = modelId.toLowerCase();
  if (GEMINI_IMAGE_HINTS.some((h) => m.includes(h))) {
    return withEnableThinkingStyle();
  }
  // Must be checked BEFORE the broader `gemini-2.5-flash` prefix.
  if (m.startsWith("gemini-2.5-flash-lite")) {
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "max",
      ] as const,
    });
  }
  if (GEMINI3_PRO_PATTERN.test(m) || GEMINI3_PRO_PREFIXES.some((p) => m.startsWith(p))) {
    // Gemini 3.x Pro cannot fully disable thinking and rejects "minimal".
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["low", "medium", "high"] as const,
    });
  }
  if (GEMINI3_FLASH_PATTERN.test(m) || GEMINI3_FLASH_PREFIXES.some((p) => m.startsWith(p))) {
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: [
        "minimal",
        "low",
        "medium",
        "high",
      ] as const,
    });
  }
  if (GEMINI25_PRO_PREFIXES.some((p) => m.startsWith(p))) {
    // Gemini 2.5 Pro rejects thinkingBudget 0, so the off switch is hidden.
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["low", "medium", "high", "max"] as const,
    });
  }
  if (GEMINI25_FLASH_PREFIXES.some((p) => m.startsWith(p))) {
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: [
        "none",
        "low",
        "medium",
        "high",
        "max",
      ] as const,
    });
  }
  return withEnableThinkingStyle();
}

function resolveMistralReasoningCapabilities(modelId: string): ExternalReasoningCapabilities {
  if (modelId === "magistral-medium-latest") {
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["medium", "high"] as const,
    });
  }
  if (modelId === "mistral-small-latest" || modelId === "mistral-vibe-cli-latest") {
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: ["none", "high"] as const,
    });
  }
  return withEnableThinkingStyle();
}

export interface ExternalReasoningResolveOptions {
  isReasoningProvider?: boolean;
  baseUrl?: string | null;
  apiType?: "chat_completions" | "responses";
  reasoningConfig?: unknown;
}

export function effectiveExternalReasoningProviderType(
  providerType: string | null | undefined,
  apiType?: "chat_completions" | "responses",
): string {
  const normalizedProvider = providerType?.trim().toLowerCase() ?? "";
  return normalizedProvider === "custom" && apiType === "responses"
    ? "openai"
    : normalizedProvider;
}

// Thinking off sends "none".
const OLLAMA_EFFORT_LEVELS = ["low", "medium", "high", "max"] as const;

// Ollama errors thinking requests on models without the "thinking" capability.
function resolveProviderReasoning(
  normalizedProvider: string,
  modelId: string,
  options: ExternalReasoningResolveOptions | undefined,
): ExternalReasoningCapabilities | null {
  if (normalizedProvider === "vllm" && options?.isReasoningProvider) {
    return withEnableThinkingStyle({
      supportsReasoning: true,
      supportsReasoningOff: true,
    });
  }
  if (
    normalizedProvider === "ollama" &&
    providerModelSupportsThinking(normalizedProvider, modelId) === true
  ) {
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: OLLAMA_EFFORT_LEVELS,
    });
  }
  return null;
}

type ReasoningWire = {
  levels: readonly ReasoningEffortLevel[] | null;
  aliases?: Partial<Record<ReasoningEffortLevel, ReasoningEffortLevel>>;
  off: "unified" | "none-level";
};

const LOCAL_SERVER_WIRE: ReasoningWire = {
  levels: ["low", "medium", "high"],
  aliases: { minimal: "low", xhigh: "high", max: "high" },
  off: "unified",
};

/** `levels: null` forwards the whole scale; `[]` means on/off only. */
const CATALOG_REASONING_WIRE: Record<string, ReasoningWire> = {
  openrouter: { levels: null, off: "unified" },
  openai: { levels: null, off: "none-level" },
  openai_codex: { levels: null, off: "none-level" },
  anthropic: { levels: null, off: "unified" },
  gemini: { levels: ["minimal", "low", "medium", "high", "xhigh", "max"], off: "unified" },
  mistral: { levels: ["high"], off: "unified" },
  kimi: { levels: [], off: "unified" },
  deepseek: { levels: ["low", "high", "max"], aliases: { minimal: "low", medium: "high", xhigh: "high" }, off: "unified" },
  qwen: { levels: [], off: "unified" },
  huggingface: { levels: null, off: "unified" },
  ollama: { levels: null, off: "unified" },
  vllm: LOCAL_SERVER_WIRE,
  llama_cpp: LOCAL_SERVER_WIRE,
};

function projectCatalogEntry(
  entry: ModelCatalogEntry,
  wire: ReasoningWire,
): ExternalReasoningCapabilities {
  if (!entry.reasoning) return withEnableThinkingStyle();
  const supportsOff =
    wire.off === "unified" ? !entry.mandatory : entry.efforts.includes("none");
  let levels: ReasoningEffortLevel[] = [];
  if (wire.levels === null || wire.levels.length > 0) {
    const mapped = entry.efforts
      .filter((level) => level !== "none")
      .map((level) => wire.aliases?.[level] ?? level)
      .filter((level) => wire.levels === null || wire.levels.includes(level));
    levels = sortReasoningEfforts(mapped);
  }
  if (levels.length === 0) {
    return withEnableThinkingStyle({
      supportsReasoning: true,
      reasoningAlwaysOn: entry.mandatory,
      supportsReasoningOff: supportsOff,
    });
  }
  const ladder: ReasoningEffortLevel[] = supportsOff ? ["none", ...levels] : levels;
  // OpenRouter reports default_effort "none" for some models, so check the final ladder.
  const mappedDefault = entry.defaultEffort
    ? (wire.aliases?.[entry.defaultEffort] ?? entry.defaultEffort)
    : null;
  const defaultEffort = mappedDefault && ladder.includes(mappedDefault) ? mappedDefault : null;
  return {
    ...withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: supportsOff,
      reasoningEffortLevels: ladder,
    }),
    // After the spread: withReasoningEffortStyle hardcodes reasoningAlwaysOn to false.
    reasoningAlwaysOn: entry.mandatory,
    defaultEffort,
  };
}

function catalogCapabilities(
  providerType: string,
  modelId: string,
  byName = false,
): ExternalReasoningCapabilities | null {
  const wire = CATALOG_REASONING_WIRE[providerType];
  if (!wire) return null;
  const entry = byName
    ? resolveModelCatalogEntryByName(modelId)
    : resolveModelCatalogEntry(providerType, modelId);
  return entry ? projectCatalogEntry(entry, wire) : null;
}

const OPENROUTER_GENERIC_TOGGLE: ExternalReasoningCapabilities = {
  supportsReasoning: true,
  reasoningStyle: "enable_thinking",
  reasoningAlwaysOn: false,
  supportsReasoningOff: true,
  reasoningEffortLevels: DEFAULT_EFFORT_LEVELS,
};

export function getExternalReasoningCapabilities(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
  options?: ExternalReasoningResolveOptions,
): ExternalReasoningCapabilities {
  // Check the connection before the catalog: known models must not opt Custom in.
  if (
    providerType?.trim().toLowerCase() === "custom" &&
    options?.apiType !== "responses"
  ) {
    const config = normalizeCustomReasoningConfig(options?.reasoningConfig);
    if (!config?.enabled) {
      return isOpenRouterMandatoryReasoningModel(modelId ?? "")
        ? withEnableThinkingStyle({
            supportsReasoning: true,
            reasoningAlwaysOn: true,
            supportsReasoningOff: false,
          })
        : withEnableThinkingStyle();
    }
    return config.style === "reasoning_effort" || config.style === "reasoning"
      ? withReasoningEffortStyle({
          supportsReasoning: true,
          supportsReasoningOff: true,
          reasoningEffortLevels: ["none", "low", "medium", "high"],
        })
      : withEnableThinkingStyle({ supportsReasoning: true, supportsReasoningOff: true });
  }
  // The capability map is keyed by the catalog's id, so look it up before case-folding.
  const catalogModel = modelId?.trim() ?? "";
  const normalizedModel = catalogModel.toLowerCase();
  const normalizedProvider = effectiveExternalReasoningProviderType(
    providerType,
    options?.apiType,
  );
  const providerLevel = resolveProviderReasoning(
    normalizedProvider,
    catalogModel,
    options,
  );
  if (providerLevel) {
    return providerLevel;
  }
  if (!normalizedModel) {
    return withEnableThinkingStyle();
  }

  if (normalizedProvider === "openrouter" && !normalizedModel.startsWith("openrouter/")) {
    const catalog = catalogCapabilities(normalizedProvider, normalizedModel);
    if (catalog) return catalog;
  }

  if (isOpenRouterMandatoryReasoningModel(normalizedModel)) {
    return withEnableThinkingStyle({
      supportsReasoning: true,
      reasoningAlwaysOn: true,
      supportsReasoningOff: false,
    });
  }

  const modelForMatching =
    normalizedProvider === "openrouter" && normalizedModel.includes("/")
      ? normalizedModel.split("/").at(-1) ?? normalizedModel
      : normalizedModel;

  switch (normalizedProvider) {
    case "openrouter": {
      // OpenRouter's `reasoning` param no-ops for non-reasoning models, so a toggle is safe.
      return OPENROUTER_GENERIC_TOGGLE;
    }
    case "kimi": {
      const table = resolveKimiReasoningCapabilities(modelForMatching);
      return table.supportsReasoning
        ? table
        : (catalogCapabilities("kimi", normalizedModel) ?? table);
    }
    case "mistral": {
      const table = resolveMistralReasoningCapabilities(modelForMatching);
      return table.supportsReasoning
        ? table
        : (catalogCapabilities("mistral", normalizedModel) ?? table);
    }
    case "gemini": {
      // Compat gateways drop native thinkingConfig, so hide the ladder.
      if (isGeminiCustomOpenAICompatBase(options?.baseUrl)) {
        return withEnableThinkingStyle();
      }
      const table = resolveGeminiReasoningCapabilities(modelForMatching);
      if (table.supportsReasoning || GEMINI_IMAGE_HINTS.some((hint) => normalizedModel.includes(hint))) {
        return table;
      }
      return catalogCapabilities("gemini", normalizedModel) ?? table;
    }
    case "ollama":
    case "deepseek":
    case "qwen":
    case "huggingface":
      return catalogCapabilities(normalizedProvider, normalizedModel) ?? withEnableThinkingStyle();
    case "vllm":
    case "llama_cpp":
      return catalogCapabilities(normalizedProvider, normalizedModel, true) ?? withEnableThinkingStyle();
    case "openai":
    case "openai_codex":
    case "anthropic": {
      const isOpenAIProvider = normalizedProvider !== "anthropic";
      const providerCaps = isOpenAIProvider
        ? resolveOpenAIReasoningEffortCapabilities(modelForMatching)
        : resolveAnthropicReasoningEffortCapabilities(modelForMatching);
      if (providerCaps.supportsReasoning) {
        return withReasoningEffortStyle(providerCaps);
      }
      if (isOpenAIProvider && OPENAI_NON_REASONING_CHAT_ALIAS.test(modelForMatching)) {
        return withEnableThinkingStyle();
      }
      return (
        catalogCapabilities(isOpenAIProvider ? "openai" : "anthropic", modelForMatching) ??
        withEnableThinkingStyle()
      );
    }
    default:
      return withEnableThinkingStyle();
  }
}

export type RuntimeReasoningFields = Pick<
  ExternalReasoningCapabilities,
  "supportsReasoning" | "reasoningAlwaysOn" | "reasoningStyle" | "supportsReasoningOff" | "reasoningEffortLevels"
> & { reasoningEffort: ReasoningEffortLevel; reasoningEnabled: boolean };

export function reasoningFieldsAfterCatalogRefresh(
  current: { reasoningEffort: ReasoningEffortLevel; reasoningEnabled: boolean },
  caps: ExternalReasoningCapabilities,
): RuntimeReasoningFields {
  const levels = caps.reasoningEffortLevels;
  return {
    supportsReasoning: caps.supportsReasoning,
    reasoningAlwaysOn: caps.reasoningAlwaysOn,
    reasoningStyle: caps.reasoningStyle,
    supportsReasoningOff: caps.supportsReasoningOff,
    reasoningEffortLevels: levels,
    reasoningEffort:
      levels.length > 0 && !levels.includes(current.reasoningEffort)
        ? clampReasoningEffortToLevels(current.reasoningEffort, levels)
        : current.reasoningEffort,
    reasoningEnabled:
      caps.supportsReasoning && !caps.supportsReasoningOff ? true : current.reasoningEnabled,
  };
}

export function providerSupportsPreserveThinking(
  providerType: string | null | undefined,
): boolean {
  return providerType === "llama_cpp";
}
