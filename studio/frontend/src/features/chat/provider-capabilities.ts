// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  normalizeProviderMaxOutputTokens,
  providerModelSupportsStudioTools,
} from "./external-providers";

/** Per-provider sampling capability matrix from each provider's chat docs (2026-05).
 *  Params a provider rejects are hidden; local models use a null capability, so all render. */

export interface ProviderCapabilities {
  /** Temperature. Reasoning-class models (gpt-5.x / o3 via /v1/responses) reject it. */
  temperature: boolean;
  /** Nucleus (top_p) sampling. Same restriction as `temperature` on OpenAI. */
  topP: boolean;
  /** top-k token sampling (only Anthropic on the providers we ship). */
  topK: boolean;
  /** min-p token cutoff (no SaaS provider currently exposes this). */
  minP: boolean;
  /** Repetition penalty (no SaaS provider currently exposes this). */
  repetitionPenalty: boolean;
  /** OpenAI-style presence penalty. */
  presencePenalty: boolean;
}

export type ExternalReasoningCapabilities = {
  supportsReasoning: boolean;
  // Mirrors the store's ReasoningStyle. "enable_thinking_effort" exists so a local model's
  // caps can be assigned here without narrowing.
  reasoningStyle: "enable_thinking" | "reasoning_effort" | "enable_thinking_effort";
  reasoningAlwaysOn: boolean;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: readonly (
    | "none"
    | "minimal"
    | "low"
    | "medium"
    | "high"
    | "max"
    | "xhigh"
  )[];
};

/** Pick a stored effort level present in `effortLevels`, mapping legacy "xhigh" to "max"
 *  when only the latter is exposed (Claude 4.6). */
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
  return effortLevels[0] ?? "low";
}

/** Fallback cap for a model with no documented limit and no connection override. */
export const EXTERNAL_MAX_OUTPUT_TOKENS = 32768;

/** Per-model max-output caps from provider docs. The local-model path is unaffected. */
const EXTERNAL_MAX_OUTPUT_TOKENS_BY_MODEL: Array<{
  providerType: string;
  prefixes: readonly string[];
  cap: number;
}> = [
  // OpenAI. The Responses API rejects an over-limit max_output_tokens rather
  // than clamping it, so an overstated cap is a failed request and an
  // understated one only a shorter answer. First match wins: a bare family
  // prefix (`gpt-5`, `gpt-4`) goes last or it swallows its own minors, and the
  // `-chat-latest` aliases go first because they cap at 16,384 whatever their
  // family does.
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
  // `chatgpt-4o-latest` shares the gpt-4o cap but not its prefix.
  { providerType: "openai", prefixes: ["gpt-4o", "chatgpt-4o"], cap: 16384 },
  // Under the 8,192 default, so these two fail on an untouched config.
  {
    providerType: "openai",
    prefixes: ["gpt-3.5-turbo", "gpt-4-turbo"],
    cap: 4096,
  },
  { providerType: "openai", prefixes: ["gpt-4"], cap: 8192 },
  // Anthropic
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
      // Dated 4.0 id, surfaced now the `-YYYYMMDD` filters are gone.
      "claude-sonnet-4-20250514",
    ],
    cap: 64000,
  },
  // Below the 32,768 fallback, so a raised Max Tokens would overshoot.
  {
    providerType: "anthropic",
    prefixes: ["claude-opus-4-1", "claude-opus-4-20250514"],
    cap: 32000,
  },
  // Gemini
  {
    providerType: "gemini",
    prefixes: ["gemini-3", "gemini-pro", "gemini-flash"],
    cap: 65536,
  },
  // DeepSeek (V4: deepseek-chat / deepseek-reasoner alias V4-flash).
  { providerType: "deepseek", prefixes: ["deepseek"], cap: 384000 },
];

/** The connection's effective Max Tokens ceiling for one model. A documented per-model cap
 *  bounds the override rather than replacing it (one connection fronts many router models).
 *  Undocumented takes the override, then EXTERNAL_MAX_OUTPUT_TOKENS; the output floor wins. */
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

/** The published per-model cap, or null when nothing documents this id. Generic Custom
 *  connections always read undocumented. OpenRouter `provider/model` prefixes are stripped. */
function _documentedMaxOutputTokens(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): number | null {
  if (!providerType || !modelId) return null;
  const normalized = modelId.trim().toLowerCase();
  if (!normalized) return null;
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

/** The lowered Max Tokens to write back, or null to leave it. The guards are load-bearing:
 *  the caller PERSISTS this and it only lowers, while maxTokensMax collapses to the 32,768
 *  fallback when the provider is unresolved. No provider means unknown, not 32,768. */
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

/** Whether the provider offers a server-side web-search tool. Enables the Search button and
 *  sends `enable_tools` + `enabled_tools: ["web_search"]` for the backend to translate.
 *  Mistral is excluded: its connector is Agents-API only and errors on /v1/chat/completions. */
export function providerSupportsBuiltinWebSearch(
  providerType: string | null | undefined,
  modelId?: string | null | undefined,
  baseUrl?: string | null | undefined,
): boolean {
  // Gemini ships grounded search on chat-capable models. Most image-tier ids reject text-tool
  // wiring, but Google documents it on the Gemini 3 image family, so allow only there.
  // Custom Gemini OpenAI-compat proxies skip the native translator, so hide the pill.
  // Output comes back inline as executableCode / codeExecutionResult parts.
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

/** Whether the provider exposes a server-side web_fetch tool emitting a document block.
 *  Anthropic-only today; gates the Fetch pill independently of Search. */
export function providerSupportsBuiltinWebFetch(
  providerType: string | null | undefined,
): boolean {
  return providerType === "anthropic";
}

/** Whether provider + model support Anthropic fast-mode (`speed: "fast"`). Opus 5 / 4.8 only:
 *  4.7 errors on `speed`, 4.6 accepts it but runs at standard speed. Backend re-checks. */
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
  // Family boundary ("" or "-") required so IDs like "claude-opus-4-70" do not match.
  return ANTHROPIC_FAST_MODE_MODEL_PREFIXES.some(
    (prefix) => modelId === prefix || modelId.startsWith(`${prefix}-`),
  );
}

/** Whether the provider/model exposes server-side code execution: Anthropic's
 *  `code_execution_20250825` sandbox, or OpenAI cloud's `shell` on /v1/responses.
 *  False elsewhere; the backend also gates the shell tool on is_openai_cloud. */
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
  // Deprecated upstream but the registry still exposes these ids; keep the pill working for users on those snapshots.
  "claude-opus-4-1",
  "claude-opus-4",
  "claude-sonnet-4",
] as const;

// OpenAI cloud shell gating: the gpt-5.6 family lists the hosted shell and gpt-5.5-pro
// shares the /v1/responses contract. Check `gpt-5.5-pro` first so the prefix match
// cannot collide with a hypothetical `gpt-5.5-turbo`.
const OPENAI_CODE_EXECUTION_MODEL_PREFIXES = [
  "gpt-5.6",
  "gpt-5.5-pro",
  "gpt-5.5",
] as const;

/** Strict check for OpenAI managed cloud or Azure Foundry, not a custom OpenAI-compat
 *  backend: shell and image-generation tools 400 elsewhere. Mirrors _is_openai_family_cloud. */
function isOpenAICloudBaseUrl(baseUrl: string | null | undefined): boolean {
  if (!baseUrl) return true; // No override → uses the default openai.com base.
  try {
    const host = new URL(baseUrl).hostname.toLowerCase();
    return host === "api.openai.com" || host.endsWith(".openai.azure.com");
  } catch {
    return false;
  }
}

export function providerSupportsBuiltinCodeExecution(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
  baseUrl?: string | null,
): boolean {
  const normalized = modelId?.trim().toLowerCase() ?? "";
  if (!normalized) return false;
  if (providerType === "anthropic") {
    return ANTHROPIC_CODE_EXECUTION_MODEL_PREFIXES.some((prefix) =>
      normalized.startsWith(prefix),
    );
  }
  if (providerType === "openai_codex") {
    return providerModelSupportsStudioTools(providerType, modelId) === true;
  }

  if (providerType === "openai") {
    if (!isOpenAICloudBaseUrl(baseUrl)) return false;
    return OPENAI_CODE_EXECUTION_MODEL_PREFIXES.some((prefix) =>
      normalized.startsWith(prefix),
    );
  }
  if (providerType === "gemini") {
    // Gemini code execution works on chat-capable models, but image-tier ids reject text-tool
    // wiring (mutually exclusive with the inline-image path), and custom OAI-compat proxies
    // skip the native translator. Wire-up lives in `_stream_gemini`.
    if (isGeminiCustomOpenAICompatBase(baseUrl)) return false;
    if (isGeminiImageModel(normalized)) return false;
    return normalized.startsWith("gemini-");
  }
  return false;
}

/** Whether the provider TYPE ships its own code sandbox, model aside. Mirrors backend
 *  PROVIDER_REGISTRY `hosted_tools` code_execution. Coarser than the per-model check:
 *  it asks whether running the code locally would be a relocation. */
const PROVIDER_TYPES_WITH_CODE_SANDBOX = new Set(["openai", "anthropic", "gemini"]);

export function providerHostsCodeExecution(
  providerType: string | null | undefined,
): boolean {
  return PROVIDER_TYPES_WITH_CODE_SANDBOX.has(providerType ?? "");
}

/** Whether provider/model exposes OpenAI's Responses-API image_generation tool. On for
 *  OpenAI cloud Responses-family ids, mirroring the backend's is_openai_cloud gate. */
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
): boolean {
  const normalized = modelId?.trim().toLowerCase() ?? "";
  if (!normalized) return false;
  if (providerType === "openai") {
    if (!isOpenAICloudBaseUrl(baseUrl)) return false;
    return OPENAI_IMAGE_GENERATION_MODEL_PREFIXES.some((prefix) =>
      normalized.startsWith(prefix),
    );
  }
  if (providerType === "gemini") {
    // Gemini Nano Banana image ids carry `-image` or the `nano-banana` alias; the backend maps
    // inlineData into the same image_b64 envelope. Custom OAI-compat proxies skip the
    // native translator, so hide the pill there.
    if (isGeminiCustomOpenAICompatBase(baseUrl)) return false;
    return normalized.includes("-image") || normalized.includes("nano-banana");
  }
  return false;
}

/** Whether `modelId` is a Gemini image-output id. Mirrors the backend's is_image_picker_model
 *  so text-only tool pills stay hidden. */
function isGeminiImageModel(modelId: string): boolean {
  const m = modelId.toLowerCase();
  return m.includes("-image") || m.includes("nano-banana");
}

/** Whether the Gemini connection points at a custom OpenAI-compat gateway. The backend routes
 *  those through /chat/completions, so native Gemini tool envelopes never reach them. */
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

/** Whether this Gemini image model supports googleSearch. Documented on the Gemini 3 image
 *  family; older ids reject it with "Search as tool is not enabled for this model". */
function geminiImageModelAllowsGoogleSearch(modelId: string): boolean {
  const m = modelId.toLowerCase();
  return (
    m.startsWith("gemini-3-pro-image") ||
    m.startsWith("gemini-3.1-flash-image") ||
    m.startsWith("nano-banana-pro") ||
    m.startsWith("nano-banana-2")
  );
}

/** Per-provider minimum outbound max_tokens. Kimi needs >= 16000 on thinking models so
 *  reasoning_content and the answer both fit; others use the generic 64. Adapter and
 *  slider min resolve the same floor. */
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

const ALL_SUPPORTED: ProviderCapabilities = {
  temperature: true,
  topP: true,
  topK: true,
  minP: true,
  repetitionPenalty: true,
  presencePenalty: true,
};

const PROVIDER_CAPABILITIES: Record<string, ProviderCapabilities> = {
  // OpenAI's flagship ids are reasoning-class on /v1/responses, which rejects temperature,
  // top_p and presence/frequency penalty.
  // The concrete ids are gpt-5.x, o3 and gpt-4.5. See external_provider._stream_openai_responses.
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
  // Anthropic accepts top_k on 3.x and 4.5/4.6, but 4.7 400s on it, so the panel surfaces it
  // and the backend strips per-model. Presence/frequency penalty is not in the Messages API.
  // Claude 4.7 is Opus, Sonnet and Haiku alike. Stripping lives in _stream_anthropic in
  // core/inference/external_provider.py.
  anthropic: {
    temperature: true,
    topP: true,
    topK: true,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
  },
  mistral: OPENAI_COMPAT_BASE,
  // Gemini's generationConfig accepts temperature, topP, topK, presencePenalty; minP and
  // repetitionPenalty are not in the contract. Shaping lives in _stream_gemini.
  // Gemini also accepts a frequencyPenalty this panel does not surface.
  gemini: {
    temperature: true,
    topP: true,
    topK: true,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: true,
  },
  // Kimi k2.5/k2.6 lock temperature and top_p and 400 on any other value, so hide both
  // sliders. The backend also strips them via PROVIDER_REGISTRY['kimi']['body_omit'].
  kimi: {
    temperature: false,
    topP: false,
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: true,
  },
  // DeepSeek deprecated presence/frequency penalty in their current docs.
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
  // OpenRouter silently drops unsupported params, so surface every knob and let the gateway
  // fan out per model.
  openrouter: ALL_SUPPORTED,
  // Local OpenAI-compat connections use the OpenAI path, but vLLM/Ollama/llama.cpp users
  // want top_k/min_p/repetition, so be permissive.
  custom: ALL_SUPPORTED,
  vllm: ALL_SUPPORTED,
  ollama: ALL_SUPPORTED,
  llama_cpp: ALL_SUPPORTED,
};

const DEFAULT_EXTERNAL_CAPABILITIES = OPENAI_COMPAT_BASE;

/** Resolve the capability set for an external provider. Null for a local model, which
 *  callers treat as "every knob applies". */
export function getProviderCapabilities(
  providerType: string | null | undefined,
): ProviderCapabilities | null {
  if (!providerType) return null;
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
    // Fable / Mythos 5 always think: `thinking.type "disabled"` 400s, so there is no off switch.
    prefixes: ["claude-fable-5", "claude-mythos-5"],
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
    // Backend maps semantic levels to manual budget_tokens.
    supportsOff: true,
    levels: ["none", "low", "medium", "high"],
  },
] as const;

function matchesModelPrefix(
  modelId: string,
  prefixes: readonly string[],
): boolean {
  return prefixes.some((prefix) => modelId.startsWith(prefix));
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
    prefixes: ["gpt-5.5-pro", "gpt-5.4-pro"],
    supportsOff: false,
    levels: ["medium", "high", "xhigh"],
  },
  {
    // gpt-5.6 (sol/terra/luna) rejects "minimal"; the ladder is the same as gpt-5.5 / gpt-5.4.
    prefixes: ["gpt-5.6", "gpt-5.5", "gpt-5.4"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "xhigh"],
  },
  {
    prefixes: ["gpt-5.3-codex"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "xhigh"],
  },
  // 5.1 replaced "minimal" with "none" and 400s on the former; 5.2 adds
  // xhigh. Both sit ahead of bare `gpt-5`, which keeps the old ladder.
  {
    prefixes: ["gpt-5.2"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "xhigh"],
  },
  // 5.1 Codex keeps reasoning mandatory: `none` 400s (openai/codex#6647).
  // `xhigh` arrived with codex-max, which sorts first or the plain codex
  // prefix swallows it.
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
  // The Codex tuning drops the lowest tier without gaining "none".
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

/** The ChatGPT-tuned aliases (`gpt-5.1-chat-latest`, Azure's `gpt-5-chat`).
 * They are non-reasoning, so `reasoning.effort` fails the whole turn with
 * "Unsupported parameter: 'reasoning.effort' is not supported with this model".
 * Matched by suffix rather than enumerated because the prefix table below keys
 * on family, and a family prefix silently swallows its own `-chat` variant. */
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
  // Kimi has a boolean thinking toggle, not an effort scale: k2.6 on by default and
  // toggleable, k2-thinking always on, k2.5 and others none.
  // k2.6 is toggled via extra_body: {thinking: {type: enabled|disabled}}.
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

// Gemini's thinking ladder: 3.x uses the string thinkingLevel (Pro rejects MINIMAL);
// 2.5 Flash/Pro use integer thinkingBudget (0=off on Flash, -1=dynamic, Pro rejects 0);
// 2.5 Flash-Lite and image ids get none. 3.x minors match by pattern so a new
// `gemini-3.6-flash` cannot fall into the 2.5 branch. Mirrors _GEMINI3_FAMILY/_GEMINI3_PRO.
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
    // Image generation; no thinking knob.
    return withEnableThinkingStyle();
  }
  // Gemini 2.5 Flash-Lite: thinkingBudget 0 = off, positive from 512. Must be checked
  // BEFORE the broader `gemini-2.5-flash` prefix.
  // The backend maps "minimal" to that 512 floor in _stream_gemini.
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
    // Gemini 3.x Pro: thinkingLevel low/medium/high; cannot fully disable, and "minimal" is rejected on Pro.
    // Refs: https://ai.google.dev/gemini-api/docs/thinking and
    // https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-1-pro.
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["low", "medium", "high"] as const,
    });
  }
  if (GEMINI3_FLASH_PATTERN.test(m) || GEMINI3_FLASH_PREFIXES.some((p) => m.startsWith(p))) {
    // Gemini 3 Flash: thinkingLevel minimal/low/medium/high. Minimal is the closest to "off" Google offers on Gemini 3.
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
    // Gemini 2.5 Pro rejects thinkingBudget 0, so the backend coerces to a small positive
    // budget and the off switch is hidden.
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["low", "medium", "high", "max"] as const,
    });
  }
  if (GEMINI25_FLASH_PREFIXES.some((p) => m.startsWith(p))) {
    // Gemini 2.5 Flash: thinkingBudget supports 0 = off cleanly.
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
      // Native reasoning model: present baseline as Medium in the UI.
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
  /** vLLM connection flagged as a reasoning model in provider config. */
  isReasoningProvider?: boolean;
  /** Provider base URL; used to detect custom Gemini OAI-compat gateways. */
  baseUrl?: string | null;
}

// vLLM has no per-model reasoning signal on OpenAI-compat, so pin via user toggle.
function resolveConnectionLevelReasoning(
  normalizedProvider: string,
  options: ExternalReasoningResolveOptions | undefined,
): ExternalReasoningCapabilities | null {
  if (normalizedProvider === "vllm" && options?.isReasoningProvider) {
    return withEnableThinkingStyle({
      supportsReasoning: true,
      supportsReasoningOff: true,
    });
  }
  return null;
}

/** Resolve external-model thinking capabilities. Per-provider resolvers do the matching;
 *  anything else defaults to no reasoning controls. */
export function getExternalReasoningCapabilities(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
  options?: ExternalReasoningResolveOptions,
): ExternalReasoningCapabilities {
  const normalizedModel = modelId?.trim().toLowerCase() ?? "";
  const normalizedProvider = providerType?.trim().toLowerCase() ?? "";
  const connectionLevel = resolveConnectionLevelReasoning(
    normalizedProvider,
    options,
  );
  if (connectionLevel) {
    return connectionLevel;
  }
  if (!normalizedModel) {
    return withEnableThinkingStyle();
  }

  // Some OpenRouter-routed ids are mandatory-reasoning and must stay on even when they
  // arrive through aliased or custom provider routes.
  if (isOpenRouterMandatoryReasoningModel(normalizedModel)) {
    return withEnableThinkingStyle({
      supportsReasoning: true,
      reasoningAlwaysOn: true,
      supportsReasoningOff: false,
    });
  }

  // OpenRouter ids are namespaced (e.g. "openai/gpt-5.5").
  const modelForMatching =
    normalizedProvider === "openrouter" && normalizedModel.includes("/")
      ? normalizedModel.split("/").at(-1) ?? normalizedModel
      : normalizedModel;

  const isOpenAIProvider =
    normalizedProvider === "openai" || normalizedProvider === "openai_codex";
  const isAnthropicProvider = normalizedProvider === "anthropic";
  const isKimiProvider = normalizedProvider === "kimi";
  const isMistralProvider = normalizedProvider === "mistral";
  const isOpenRouterProvider = normalizedProvider === "openrouter";
  if (isOpenRouterProvider) {
    // OpenRouter's unified `reasoning` param is accepted everywhere and no-ops for non-reasoning
    // models, so past the mandatory guard everything gets a toggleable control.
    return {
      supportsReasoning: true,
      reasoningStyle: "enable_thinking",
      reasoningAlwaysOn: false,
      supportsReasoningOff: true,
      reasoningEffortLevels: DEFAULT_EFFORT_LEVELS,
    };
  }
  if (isKimiProvider) return resolveKimiReasoningCapabilities(modelForMatching);
  if (isMistralProvider) return resolveMistralReasoningCapabilities(modelForMatching);
  if (normalizedProvider === "gemini") {
    // Custom Gemini OAI-compat gateways route through /chat/completions, which drops the native
    // thinkingConfig payload, so hide the ladder.
    if (isGeminiCustomOpenAICompatBase(options?.baseUrl)) {
      return withEnableThinkingStyle();
    }
    return resolveGeminiReasoningCapabilities(modelForMatching);
  }
  if (!isOpenAIProvider && !isAnthropicProvider) {
    return withEnableThinkingStyle();
  }

  const providerCaps = isOpenAIProvider
    ? resolveOpenAIReasoningEffortCapabilities(modelForMatching)
    : resolveAnthropicReasoningEffortCapabilities(modelForMatching);
  if (providerCaps.supportsReasoning) {
    return withReasoningEffortStyle(providerCaps);
  }

  return withEnableThinkingStyle();
}
