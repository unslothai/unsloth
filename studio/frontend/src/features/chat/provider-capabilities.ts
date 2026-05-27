// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-provider sampling capability matrix sourced from each provider's
// chat-completion docs (2026-05); panel hides params the active
// provider would silently drop or reject.
// New knobs: default to false on every SaaS bucket; only local +
// openrouter expose llama.cpp samplers.
export interface ProviderCapabilities {
  /** OpenAI gpt-5.x / o-series reject via /v1/responses. */
  temperature: boolean;
  topP: boolean;
  /** Anthropic only among SaaS providers. */
  topK: boolean;
  minP: boolean;
  repetitionPenalty: boolean;
  presencePenalty: boolean;
  /** OAI Chat only; rejected by Responses + Anthropic. */
  frequencyPenalty: boolean;
  /** OAI Chat + OAI-compat. Responses + Anthropic drop. */
  seed: boolean;
  /** Not accepted by Responses; mapped to `stop_sequences` on Anthropic. */
  stop: boolean;
  /** Per-provider enum, see getServiceTierOptions. */
  serviceTier: boolean;
  /** Anthropic inverts to `disable_parallel_tool_use`. */
  parallelToolCalls: boolean;
  /** llama.cpp `typ_p`. */
  typicalP: boolean;
  /** llama.cpp `top_n_sigma`. */
  topNSigma: boolean;
  /** llama.cpp `repeat_last_n`. */
  repeatLastN: boolean;
  /** llama.cpp `dynatemp_range`. */
  dynatempRange: boolean;
  /** llama.cpp `dynatemp_exponent`. */
  dynatempExponent: boolean;
  /** llama.cpp `mirostat` (0/1/2). */
  mirostat: boolean;
  mirostatTau: boolean;
  mirostatEta: boolean;
  /** OpenRouter `top_a`. https://openrouter.ai/docs/api/reference/parameters */
  topA: boolean;
  /** llama.cpp DRY (4 fields). dryMultiplier is the master switch. */
  dryMultiplier: boolean;
  dryBase: boolean;
  dryAllowedLength: boolean;
  dryPenaltyLastN: boolean;
  /** llama.cpp XTC (2 fields). xtcProbability is the master switch. */
  xtcProbability: boolean;
  xtcThreshold: boolean;
  /** llama.cpp `min_keep`. */
  minKeep: boolean;
  /** llama.cpp + vLLM. Ollama OAI translator drops it. */
  ignoreEos: boolean;
  /** llama.cpp + vLLM. Ollama OAI translator drops it. */
  minTokens: boolean;
  /** vLLM only. */
  skipSpecialTokens: boolean;
  spacesBetweenSpecialTokens: boolean;
  /** vLLM only. Useful for agentic tools. */
  includeStopStrInOutput: boolean;
  /** vLLM only. Left-truncate the prompt. */
  truncatePromptTokens: boolean;
  /** llama.cpp `n_keep` / `n_probs`. */
  nKeep: boolean;
  nProbs: boolean;
  /** llama.cpp `cache_prompt`. */
  cachePrompt: boolean;
  /** llama.cpp debug flags. */
  returnTokens: boolean;
  timingsPerToken: boolean;
  postSamplingProbs: boolean;
}

/**
 * Per-provider stop-sequence max count. Mirrors backend `stop_max`.
 *   openai 4 (Chat hard cap; Responses drops stop)
 *   anthropic 16 (client-side guard, docs no max)
 *   kimi 5 (https://platform.kimi.ai/docs/api/chat)
 *   deepseek 16 (https://api-docs.deepseek.com/api/create-chat-completion)
 *   mistral 16, gemini 4, openrouter 4, default 16 (ollama/vllm/llama.cpp/custom)
 */
const PROVIDER_STOP_MAX: Record<string, number> = {
  openai: 4,
  anthropic: 16,
  kimi: 5,
  deepseek: 16,
  mistral: 16,
  gemini: 4,
  openrouter: 4,
};

export function getProviderStopMax(
  providerType: string | null | undefined,
): number {
  if (!providerType) return 16; // local backends
  return PROVIDER_STOP_MAX[providerType] ?? 16;
}

export type ServiceTierOption =
  | "auto"
  | "default"
  | "flex"
  | "priority"
  | "scale"
  | "standard_only";

/**
 * Legal `service_tier` per provider. anthropic=auto|standard_only;
 * openai (/v1/responses)=auto|default|flex|priority (scale excluded
 * though SDK lists it); others fall through to auto|default.
 */
export function getServiceTierOptions(
  providerType: string | null | undefined,
): readonly ServiceTierOption[] {
  if (providerType === "anthropic") {
    return ["auto", "standard_only"] as const;
  }
  if (providerType === "openai") {
    return ["auto", "default", "flex", "priority"] as const;
  }
  return ["auto", "default"] as const;
}

export type ExternalReasoningCapabilities = {
  supportsReasoning: boolean;
  reasoningStyle: "enable_thinking" | "reasoning_effort";
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

/**
 * Prefer a stored reasoning effort level that exists in ``effortLevels``,
 * mapping legacy "xhigh" to "max" when the model only exposes the latter
 * (Claude 4.6 adaptive thinking).
 */
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

/**
 * Fallback cap for unknown providers / models. Prefer
 * `getExternalMaxOutputTokens(providerType, modelId)` for the real cap.
 */
export const EXTERNAL_MAX_OUTPUT_TOKENS = 32768;

// Per-model max-output caps (verified May 2026). Longer prefixes first
// so .startsWith() picks the specific id over the family root.
//   OpenAI:    developers.openai.com/api/docs/models/<model>
//   Anthropic: platform.claude.com/docs/en/about-claude/models/overview
//   Gemini:    ai.google.dev/gemini-api/docs/models
//   DeepSeek:  api-docs.deepseek.com/quick_start/pricing
const EXTERNAL_MAX_OUTPUT_TOKENS_BY_MODEL: Array<{
  providerType: string;
  prefixes: readonly string[];
  cap: number;
}> = [
  { providerType: "openai", prefixes: ["gpt-5.3-chat-latest", "gpt-5.1-chat"], cap: 16384 },
  { providerType: "openai", prefixes: ["gpt-5"], cap: 128000 },
  { providerType: "openai", prefixes: ["o1", "o3", "o4", "codex-mini"], cap: 100000 },
  // Anthropic Opus 4.6 + 4.7 ship 128k Max output; Sonnet 4.5/4.6/4 +
  // Opus 4.5 + Haiku 4.5 ship 64k; Opus 4.1 / Opus 4 fall through to
  // the 32k EXTERNAL_MAX_OUTPUT_TOKENS default.
  {
    providerType: "anthropic",
    prefixes: ["claude-opus-4-7", "claude-opus-4-6"],
    cap: 128000,
  },
  {
    providerType: "anthropic",
    prefixes: [
      "claude-sonnet-4-6",
      "claude-opus-4-5",
      "claude-sonnet-4-5",
      "claude-haiku-4-5",
      "claude-sonnet-4",
    ],
    cap: 64000,
  },
  {
    providerType: "gemini",
    prefixes: ["gemini-3", "gemini-pro", "gemini-flash"],
    cap: 65536,
  },
  // V4: deepseek-chat / deepseek-reasoner alias V4-flash.
  { providerType: "deepseek", prefixes: ["deepseek"], cap: 384000 },
];

/**
 * Documented per-model output cap; unknown ids fall back to
 * `EXTERNAL_MAX_OUTPUT_TOKENS` (32k). OpenRouter ids are
 * `provider/model`; the prefix is stripped before matching.
 */
export function getExternalMaxOutputTokens(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): number {
  if (!providerType || !modelId) return EXTERNAL_MAX_OUTPUT_TOKENS;
  const normalized = modelId.trim().toLowerCase();
  if (!normalized) return EXTERNAL_MAX_OUTPUT_TOKENS;
  const stripped =
    providerType === "openrouter" && normalized.includes("/")
      ? normalized.split("/").slice(-1)[0]
      : normalized;
  const effectiveProvider =
    providerType === "openrouter"
      ? _inferProviderFromOpenrouterId(normalized) ?? providerType
      : providerType;
  for (const entry of EXTERNAL_MAX_OUTPUT_TOKENS_BY_MODEL) {
    if (entry.providerType !== effectiveProvider) continue;
    if (entry.prefixes.some((prefix) => stripped.startsWith(prefix))) {
      return entry.cap;
    }
  }
  return EXTERNAL_MAX_OUTPUT_TOKENS;
}

function _inferProviderFromOpenrouterId(
  normalizedId: string,
): string | null {
  // Map OpenRouter `provider/model` prefix to our internal providerType.
  if (normalizedId.startsWith("openai/")) return "openai";
  if (normalizedId.startsWith("anthropic/")) return "anthropic";
  if (normalizedId.startsWith("google/")) return "gemini";
  if (normalizedId.startsWith("deepseek/")) return "deepseek";
  return null;
}

// Gates the composer's Search button. Backend maps
// enable_tools:["web_search"] to each provider's tool schema:
//   OpenAI: tools:[{type:"web_search"}] on /v1/responses
//   Anthropic: tools:[{type:"web_search_20250305", max_uses:5}]
//   OpenRouter: plugins:[{id:"web"}]
//   Kimi: $web_search builtin (2-call via _stream_kimi_web_search)
// Mistral excluded (web_search lives on Agents API, 400s on /v1/chat).
export function providerSupportsBuiltinWebSearch(
  providerType: string | null | undefined,
  modelId?: string | null | undefined,
  baseUrl?: string | null | undefined,
): boolean {
  // Gemini ships grounded search via `tools: [{googleSearch: {}}]` on
  // every chat-capable model. Most image-tier ids (`-image`,
  // `nano-banana`) reject text-tool wiring because the
  // responseModalities path is mutually exclusive with text tools, but
  // Google explicitly documents Search grounding on the Gemini 3 image
  // family (gemini-3-pro-image-preview, gemini-3.1-flash-image-preview,
  // nano-banana-pro). Allow Search on those; hide on older image ids.
  // Custom Gemini OpenAI-compat proxies (non-Google bases) skip the
  // native translator on the backend, so native tool envelopes never
  // reach them -- hide the pill there.
  if (providerType === "gemini") {
    if (isGeminiCustomOpenAICompatBase(baseUrl)) return false;
    const normalized = modelId?.trim().toLowerCase() ?? "";
    if (normalized && isGeminiImageModel(normalized)) {
      return geminiImageModelAllowsGoogleSearch(normalized);
    }
    return true;
  }
  return (
    providerType === "openai" ||
    providerType === "anthropic" ||
    providerType === "openrouter" ||
    providerType === "kimi"
  );
}

// Anthropic-only server-side web_fetch tool
// (web_fetch_20250910 / _20260209). Gates the composer's Fetch pill.
export function providerSupportsBuiltinWebFetch(
  providerType: string | null | undefined,
): boolean {
  return providerType === "anthropic";
}

// Anthropic fast-mode (`speed:"fast"` + fast-mode-2026-02-01 header).
// Opus 4.6 / 4.7 only per
// https://platform.claude.com/docs/en/build-with-claude/fast-mode.
// Backend silently drops on unsupported models as a second defence.
const ANTHROPIC_FAST_MODE_MODEL_PREFIXES = [
  "claude-opus-4-7",
  "claude-opus-4-6",
] as const;

export function providerSupportsFastMode(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): boolean {
  if (providerType !== "anthropic") return false;
  if (!modelId) return false;
  // Family boundary required so "claude-opus-4-70" doesn't match.
  return ANTHROPIC_FAST_MODE_MODEL_PREFIXES.some(
    (prefix) => modelId === prefix || modelId.startsWith(`${prefix}-`),
  );
}

// Server-side code-execution tools:
//   Anthropic code_execution_20250825 (Python + bash + str_replace in
//     a 5 GB sandbox).
//   OpenAI cloud `shell` on /v1/responses (bash in a reusable container
//     referenced via openaiCodeExecContainerId across turns).
// Backend also gates OpenAI on is_openai_cloud so custom OAI-compat
// servers reporting provider_type="openai" can't accidentally get the
// shell tool. File uploads (container_upload / input_file) are
// follow-up work.
const ANTHROPIC_CODE_EXECUTION_MODEL_PREFIXES = [
  "claude-opus-4-7",
  "claude-opus-4-6",
  "claude-sonnet-4-6",
  "claude-opus-4-5",
  "claude-sonnet-4-5",
  "claude-haiku-4-5",
  // Deprecated upstream but the registry still exposes the ids, so the
  // pill should remain functional for users on those snapshots.
  "claude-opus-4-1",
  "claude-opus-4",
  "claude-sonnet-4",
] as const;

// OpenAI cloud shell-tool gating. Docs only explicitly demonstrate
// gpt-5.5; gpt-5.5-pro is included because the family share the same
// /v1/responses contract. `gpt-5.5-pro` is checked first so the prefix
// match doesn't collide with a hypothetical `gpt-5.5-turbo` etc.
const OPENAI_CODE_EXECUTION_MODEL_PREFIXES = [
  "gpt-5.5-pro",
  "gpt-5.5",
] as const;

/**
 * Strict check that a provider configuration points at OpenAI's
 * managed cloud (api.openai.com) or Azure OpenAI Foundry
 * (*.openai.azure.com), as opposed to a custom OpenAI-compat backend
 * (ollama / llama.cpp / vLLM / generic "custom" preset). The shell and
 * image-generation tools only exist on cloud backends; sending them to
 * anything else 400s the request. Mirror of the backend's
 * `_is_openai_family_cloud` host check.
 */
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
  if (providerType === "openai") {
    if (!isOpenAICloudBaseUrl(baseUrl)) return false;
    return OPENAI_CODE_EXECUTION_MODEL_PREFIXES.some((prefix) =>
      normalized.startsWith(prefix),
    );
  }
  if (providerType === "gemini") {
    // Gemini's `tools: [{codeExecution: {}}]` is supported on every
    // chat-capable model. Image-tier ids (`-image`, `nano-banana`)
    // reject text-tool wiring because the inline-image path is
    // mutually exclusive with codeExecution. Custom Gemini
    // OpenAI-compat proxies skip the native translator on the
    // backend, so native codeExecution envelopes do not reach them.
    // Wire-up lives in `_stream_gemini` on the backend; output comes
    // back inline as executableCode/codeExecutionResult parts. See
    // https://ai.google.dev/gemini-api/docs/code-execution.
    if (isGeminiCustomOpenAICompatBase(baseUrl)) return false;
    if (isGeminiImageModel(normalized)) return false;
    return normalized.startsWith("gemini-");
  }
  return false;
}

// OpenAI Responses-API image_generation. OpenAI cloud +
// Responses-family ids only; backend mirrors via is_openai_cloud.
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
    // Gemini's Nano Banana image-output ids carry either `-image` (e.g.
    // `gemini-2.5-flash-image`, `gemini-3.1-flash-image-preview`) or the
    // `nano-banana` alias (`nano-banana-pro-preview`). The backend flips
    // generationConfig.responseModalities to ["TEXT", "IMAGE"] when one
    // is picked, and translates inlineData parts into the same image_b64
    // tool_end envelope the OpenAI path emits so the chat UI renders the
    // picture inline. Custom Gemini OpenAI-compat proxies skip the
    // native translator on the backend, so hide the image pill there.
    // See https://ai.google.dev/gemini-api/docs/image-generation.
    if (isGeminiCustomOpenAICompatBase(baseUrl)) return false;
    return normalized.includes("-image") || normalized.includes("nano-banana");
  }
  return false;
}

/**
 * Whether `modelId` is a Gemini image-output id (Nano Banana family).
 * Mirrors the backend's `is_image_picker_model` guard so the frontend
 * hides text-only tool pills (web_search, code_execution) for these.
 */
function isGeminiImageModel(modelId: string): boolean {
  const m = modelId.toLowerCase();
  return m.includes("-image") || m.includes("nano-banana");
}

/**
 * Whether the saved Gemini connection points at a custom
 * OpenAI-compatible gateway (any non-Google host). The backend
 * `_is_openai_compatible` mirrors this to route those connections
 * through `/chat/completions` instead of the native translator, so
 * native Gemini tool envelopes (googleSearch, codeExecution,
 * responseModalities) never reach them. Hide the corresponding
 * Studio pills here so the request, builder, and UI agree.
 */
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

/**
 * Whether the given Gemini image model supports `tools: [{googleSearch: {}}]`.
 * Google documents Search grounding on the Gemini 3 image family
 * (gemini-3-pro-image-preview, gemini-3.1-flash-image-preview,
 * "Nano Banana Pro"); older image ids (gemini-2.5-flash-image) reject
 * it with "Search as tool is not enabled for this model".
 */
function geminiImageModelAllowsGoogleSearch(modelId: string): boolean {
  const m = modelId.toLowerCase();
  return (
    m.startsWith("gemini-3-pro-image") ||
    m.startsWith("gemini-3.1-flash-image") ||
    m.startsWith("nano-banana-pro") ||
    m.startsWith("nano-banana-2")
  );
}

// Per-provider min on outbound max_tokens. Kimi thinking needs >=16000
// (truncates mid-stream below); chat-adapter bumps on send.
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
  frequencyPenalty: true,
  seed: true,
  stop: true,
  serviceTier: false,
  parallelToolCalls: true,
  typicalP: false,
  topNSigma: false,
  repeatLastN: false,
  dynatempRange: false,
  dynatempExponent: false,
  mirostat: false,
  mirostatTau: false,
  mirostatEta: false,
  topA: false,
  dryMultiplier: false,
  dryBase: false,
  dryAllowedLength: false,
  dryPenaltyLastN: false,
  xtcProbability: false,
  xtcThreshold: false,
  minKeep: false,
  ignoreEos: false,
  minTokens: false,
  skipSpecialTokens: false,
  spacesBetweenSpecialTokens: false,
  includeStopStrInOutput: false,
  truncatePromptTokens: false,
  nKeep: false,
  nProbs: false,
  cachePrompt: false,
  returnTokens: false,
  timingsPerToken: false,
  postSamplingProbs: false,
};

// Unsloth's first-party llama-server runtime (provider type `llama_cpp`)
// plus the permissive `custom` preset. Exposes the full llama.cpp
// sampler chain (typical_p / top_n_sigma / mirostat / dynatemp /
// repeat_last_n) per the upstream server README. Not used for vLLM or
// Ollama — see VLLM_OLLAMA_CAPABILITIES below.
const LLAMA_CPP_CAPABILITIES: ProviderCapabilities = {
  temperature: true,
  topP: true,
  topK: true,
  minP: true,
  repetitionPenalty: true,
  presencePenalty: true,
  frequencyPenalty: true,
  seed: true,
  stop: true,
  serviceTier: false,
  parallelToolCalls: true,
  typicalP: true,
  topNSigma: true,
  repeatLastN: true,
  dynatempRange: true,
  dynatempExponent: true,
  mirostat: true,
  mirostatTau: true,
  mirostatEta: true,
  topA: false,
  dryMultiplier: true,
  dryBase: true,
  dryAllowedLength: true,
  dryPenaltyLastN: true,
  xtcProbability: true,
  xtcThreshold: true,
  minKeep: true,
  ignoreEos: true,
  minTokens: true,
  skipSpecialTokens: false,
  spacesBetweenSpecialTokens: false,
  includeStopStrInOutput: false,
  truncatePromptTokens: false,
  nKeep: true,
  nProbs: true,
  cachePrompt: true,
  returnTokens: true,
  timingsPerToken: true,
  postSamplingProbs: true,
};

// vLLM SamplingParams: OAI subset + top_k/min_p/repetition_penalty/seed
// + the 4 vLLM-only output-shape knobs. No DRY / XTC / mirostat /
// dynatemp / typical_p / min_keep / n_keep / n_probs / cache_prompt /
// debug flags (none in SamplingParams).
const VLLM_CAPABILITIES: ProviderCapabilities = {
  ...LLAMA_CPP_CAPABILITIES,
  typicalP: false,
  topNSigma: false,
  repeatLastN: false,
  dynatempRange: false,
  dynatempExponent: false,
  mirostat: false,
  mirostatTau: false,
  mirostatEta: false,
  dryMultiplier: false,
  dryBase: false,
  dryAllowedLength: false,
  dryPenaltyLastN: false,
  xtcProbability: false,
  xtcThreshold: false,
  minKeep: false,
  skipSpecialTokens: true,
  spacesBetweenSpecialTokens: true,
  includeStopStrInOutput: true,
  truncatePromptTokens: true,
  nKeep: false,
  nProbs: false,
  cachePrompt: false,
  returnTokens: false,
  timingsPerToken: false,
  postSamplingProbs: false,
};

// Ollama OAI translator (openai/openai.go FromChatRequest) only copies
// the documented OpenAI subset on /v1/chat/completions — top_k / min_p
// / repetition_penalty / ignore_eos / min_tokens / the 4 vLLM output
// knobs all silently drop on this path. (Native /api/chat would forward
// them via `options`, but Studio uses /v1.)
const OLLAMA_CAPABILITIES: ProviderCapabilities = {
  ...VLLM_CAPABILITIES,
  topK: false,
  minP: false,
  repetitionPenalty: false,
  ignoreEos: false,
  minTokens: false,
  skipSpecialTokens: false,
  spacesBetweenSpecialTokens: false,
  includeStopStrInOutput: false,
  truncatePromptTokens: false,
};

// OpenRouter is a router-of-routers: gateway accepts a wider set of
// OAI-style fields than any single upstream and silently drops what
// the chosen route doesn't. Surface the full documented set (incl.
// top_a) and leave llama.cpp-only knobs off.
// https://openrouter.ai/docs/api/reference/parameters
const OPENROUTER_CAPABILITIES: ProviderCapabilities = {
  temperature: true,
  topP: true,
  topK: true,
  minP: true,
  repetitionPenalty: true,
  presencePenalty: true,
  frequencyPenalty: true,
  seed: true,
  stop: true,
  serviceTier: false,
  parallelToolCalls: true,
  typicalP: false,
  topNSigma: false,
  repeatLastN: false,
  dynatempRange: false,
  dynatempExponent: false,
  mirostat: false,
  mirostatTau: false,
  mirostatEta: false,
  topA: true,
  dryMultiplier: false,
  dryBase: false,
  dryAllowedLength: false,
  dryPenaltyLastN: false,
  xtcProbability: false,
  xtcThreshold: false,
  minKeep: false,
  ignoreEos: false,
  minTokens: false,
  skipSpecialTokens: false,
  spacesBetweenSpecialTokens: false,
  includeStopStrInOutput: false,
  truncatePromptTokens: false,
  nKeep: false,
  nProbs: false,
  cachePrompt: false,
  returnTokens: false,
  timingsPerToken: false,
  postSamplingProbs: false,
};

// OpenAI reasoning class via /v1/responses: temperature fixed at 1,
// top_p ignored, 400s on presence/frequency_penalty/seed. Chat-class
// (gpt-4o etc) keeps the full surface even via /v1/responses. Both
// drop `stop` (Responses doesn't surface it).
// https://platform.openai.com/docs/guides/reasoning
const OPENAI_REASONING_CAPABILITIES: ProviderCapabilities = {
  temperature: false,
  topP: false,
  topK: false,
  minP: false,
  repetitionPenalty: false,
  presencePenalty: false,
  frequencyPenalty: false,
  seed: false,
  stop: false,
  serviceTier: true,
  parallelToolCalls: true,
  typicalP: false,
  topNSigma: false,
  repeatLastN: false,
  dynatempRange: false,
  dynatempExponent: false,
  mirostat: false,
  mirostatTau: false,
  mirostatEta: false,
  topA: false,
  dryMultiplier: false,
  dryBase: false,
  dryAllowedLength: false,
  dryPenaltyLastN: false,
  xtcProbability: false,
  xtcThreshold: false,
  minKeep: false,
  ignoreEos: false,
  minTokens: false,
  skipSpecialTokens: false,
  spacesBetweenSpecialTokens: false,
  includeStopStrInOutput: false,
  truncatePromptTokens: false,
  nKeep: false,
  nProbs: false,
  cachePrompt: false,
  returnTokens: false,
  timingsPerToken: false,
  postSamplingProbs: false,
};
const OPENAI_CHAT_CAPABILITIES: ProviderCapabilities = {
  temperature: true,
  topP: true,
  topK: false,
  minP: false,
  repetitionPenalty: false,
  presencePenalty: true,
  frequencyPenalty: true,
  seed: true,
  // Responses API does not surface `stop` even for non-reasoning models;
  // tracked in OpenAI's Responses-vs-ChatCompletions migration notes.
  stop: false,
  serviceTier: true,
  parallelToolCalls: true,
  typicalP: false,
  topNSigma: false,
  repeatLastN: false,
  dynatempRange: false,
  dynatempExponent: false,
  mirostat: false,
  mirostatTau: false,
  mirostatEta: false,
  topA: false,
  dryMultiplier: false,
  dryBase: false,
  dryAllowedLength: false,
  dryPenaltyLastN: false,
  xtcProbability: false,
  xtcThreshold: false,
  minKeep: false,
  ignoreEos: false,
  minTokens: false,
  skipSpecialTokens: false,
  spacesBetweenSpecialTokens: false,
  includeStopStrInOutput: false,
  truncatePromptTokens: false,
  nKeep: false,
  nProbs: false,
  cachePrompt: false,
  returnTokens: false,
  timingsPerToken: false,
  postSamplingProbs: false,
};

// Prefix list for OpenAI reasoning-class model ids. Kept in sync with
// OPENAI_REASONING_MODELS below (used for reasoning_effort capability).
// Longest prefixes first so "gpt-5.5-pro" wins over "gpt-5.5".
const OPENAI_REASONING_MODEL_PREFIXES = [
  "gpt-5.5-pro",
  "gpt-5.5",
  "gpt-5.4-pro",
  "gpt-5.4",
  "gpt-5.3-chat-latest",
  "gpt-5.3-codex",
  "gpt-5.3",
  "gpt-5.2",
  "gpt-5.1",
  "gpt-5",
  "o1",
  "o3",
  "o4",
] as const;

function isOpenAIReasoningModelId(modelId: string | null | undefined): boolean {
  const normalized = modelId?.trim().toLowerCase() ?? "";
  if (!normalized) return false;
  return OPENAI_REASONING_MODEL_PREFIXES.some((p) => normalized.startsWith(p));
}

// Mirror of backend _ANTHROPIC_4_7_SAMPLING_REMOVED. Opus 4.7 removed
// temperature/top_p/top_k; only Opus shipped in 4.7. The -4-7[-.]/EOL
// anchor keeps future families (claude-opus-5 etc) unaffected.
const ANTHROPIC_4_7_SAMPLING_REMOVED_REGEX = /^claude-opus-4-7(?:[-.]|$)/i;

function isClaude47SamplingRemoved(modelId: string | null | undefined): boolean {
  const normalized = modelId?.trim().toLowerCase() ?? "";
  if (!normalized) return false;
  return ANTHROPIC_4_7_SAMPLING_REMOVED_REGEX.test(normalized);
}

// DeepSeek reasoner ids silently ignore temperature/top_p/presence/
// frequency and 400 on logprobs per the reasoning_model guide. Prefix
// match covers future revisions (deepseek-reasoner-2027 etc).
const DEEPSEEK_REASONING_MODEL_PREFIXES = [
  "deepseek-reasoner",
  "deepseek-r1",
] as const;

function isDeepSeekReasoningModelId(modelId: string | null | undefined): boolean {
  const normalized = modelId?.trim().toLowerCase() ?? "";
  if (!normalized) return false;
  return DEEPSEEK_REASONING_MODEL_PREFIXES.some((p) => normalized.startsWith(p));
}

const PROVIDER_CAPABILITIES: Record<string, ProviderCapabilities> = {
  // Default to reasoning-class; getProviderCapabilities upgrades
  // non-reasoning ids (gpt-4o etc) to OPENAI_CHAT_CAPABILITIES.
  openai: OPENAI_REASONING_CAPABILITIES,
  // Messages API: temperature/top_p/top_k/stop_sequences/service_tier
  // (auto|standard_only)/disable_parallel_tool_use. Opus 4.7 strips
  // temperature/top_p/top_k via the regex above. No presence/frequency
  // penalty / seed / logprobs on any Claude generation.
  anthropic: {
    temperature: true,
    topP: true,
    topK: true,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
    frequencyPenalty: false,
    seed: false,
    stop: true,
    serviceTier: true,
    parallelToolCalls: true,
    typicalP: false,
    topNSigma: false,
    repeatLastN: false,
    dynatempRange: false,
    dynatempExponent: false,
    mirostat: false,
    mirostatTau: false,
    mirostatEta: false,
    topA: false,
    dryMultiplier: false,
    dryBase: false,
    dryAllowedLength: false,
    dryPenaltyLastN: false,
    xtcProbability: false,
    xtcThreshold: false,
    minKeep: false,
    ignoreEos: false,
    minTokens: false,
    skipSpecialTokens: false,
    spacesBetweenSpecialTokens: false,
    includeStopStrInOutput: false,
    truncatePromptTokens: false,
    nKeep: false,
    nProbs: false,
    cachePrompt: false,
    returnTokens: false,
    timingsPerToken: false,
    postSamplingProbs: false,
  },
  mistral: OPENAI_COMPAT_BASE,
  // Gemini generationConfig: temperature/topP/topK/presencePenalty/
  // frequencyPenalty/seed/stopSequences. No minP/repetitionPenalty.
  // https://ai.google.dev/api/rest/v1beta/GenerationConfig
  gemini: {
    temperature: true,
    topP: true,
    topK: true,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: true,
    frequencyPenalty: false,
    seed: false,
    stop: true,
    serviceTier: false,
    parallelToolCalls: false,
    typicalP: false,
    topNSigma: false,
    repeatLastN: false,
    dynatempRange: false,
    dynatempExponent: false,
    mirostat: false,
    mirostatTau: false,
    mirostatEta: false,
    topA: false,
    dryMultiplier: false,
    dryBase: false,
    dryAllowedLength: false,
    dryPenaltyLastN: false,
    xtcProbability: false,
    xtcThreshold: false,
    minKeep: false,
    ignoreEos: false,
    minTokens: false,
    skipSpecialTokens: false,
    spacesBetweenSpecialTokens: false,
    includeStopStrInOutput: false,
    truncatePromptTokens: false,
    nKeep: false,
    nProbs: false,
    cachePrompt: false,
    returnTokens: false,
    timingsPerToken: false,
    postSamplingProbs: false,
  },
  // Kimi K2.x locks temperature + top_p ("only 1 is allowed for this
  // model"); seed + parallel_tool_calls aren't in the Chat schema
  // (platform.kimi.ai/docs/api/chat). Backend strips via body_omit.
  kimi: {
    temperature: false,
    topP: false,
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: true,
    // K2.x 400s on non-default frequency_penalty; backend strips too.
    frequencyPenalty: false,
    seed: false,
    stop: true,
    serviceTier: false,
    parallelToolCalls: false,
    typicalP: false,
    topNSigma: false,
    repeatLastN: false,
    dynatempRange: false,
    dynatempExponent: false,
    mirostat: false,
    mirostatTau: false,
    mirostatEta: false,
    topA: false,
    dryMultiplier: false,
    dryBase: false,
    dryAllowedLength: false,
    dryPenaltyLastN: false,
    xtcProbability: false,
    xtcThreshold: false,
    minKeep: false,
    ignoreEos: false,
    minTokens: false,
    skipSpecialTokens: false,
    spacesBetweenSpecialTokens: false,
    includeStopStrInOutput: false,
    truncatePromptTokens: false,
    nKeep: false,
    nProbs: false,
    cachePrompt: false,
    returnTokens: false,
    timingsPerToken: false,
    postSamplingProbs: false,
  },
  // DeepSeek schema (api-docs.deepseek.com/api/create-chat-completion)
  // lists temperature/top_p/stop only — no seed or parallel_tool_calls.
  // Presence/frequency are deprecated. Reasoner ids additionally ignore
  // temperature/top_p; getProviderCapabilities downshifts them.
  deepseek: {
    temperature: true,
    topP: true,
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
    frequencyPenalty: false,
    seed: false,
    stop: true,
    serviceTier: false,
    parallelToolCalls: false,
    typicalP: false,
    topNSigma: false,
    repeatLastN: false,
    dynatempRange: false,
    dynatempExponent: false,
    mirostat: false,
    mirostatTau: false,
    mirostatEta: false,
    topA: false,
    dryMultiplier: false,
    dryBase: false,
    dryAllowedLength: false,
    dryPenaltyLastN: false,
    xtcProbability: false,
    xtcThreshold: false,
    minKeep: false,
    ignoreEos: false,
    minTokens: false,
    skipSpecialTokens: false,
    spacesBetweenSpecialTokens: false,
    includeStopStrInOutput: false,
    truncatePromptTokens: false,
    nKeep: false,
    nProbs: false,
    cachePrompt: false,
    returnTokens: false,
    timingsPerToken: false,
    postSamplingProbs: false,
  },
  qwen: OPENAI_COMPAT_BASE,
  huggingface: OPENAI_COMPAT_BASE,
  openrouter: OPENROUTER_CAPABILITIES,
  // llama_cpp + custom: first-party llama-server, full chain.
  // vllm: OAI subset + top_k/min_p/repetition_penalty/seed.
  // ollama: stricter — OAI translator drops top_k/min_p/rep_pen too.
  custom: LLAMA_CPP_CAPABILITIES,
  llama_cpp: LLAMA_CPP_CAPABILITIES,
  vllm: VLLM_CAPABILITIES,
  ollama: OLLAMA_CAPABILITIES,
};

const DEFAULT_EXTERNAL_CAPABILITIES = OPENAI_COMPAT_BASE;

// Per-model overrides: openai+chat-class -> OPENAI_CHAT_CAPABILITIES;
// anthropic claude-opus-4-7 strips temp/top_p/top_k; deepseek reasoner
// hides temp/top_p. Local (no providerType) returns null = every knob.
export function getProviderCapabilities(
  providerType: string | null | undefined,
  modelId?: string | null | undefined,
): ProviderCapabilities | null {
  if (!providerType) return null;
  const base = PROVIDER_CAPABILITIES[providerType] ?? DEFAULT_EXTERNAL_CAPABILITIES;
  if (providerType === "openai" && modelId && !isOpenAIReasoningModelId(modelId)) {
    return OPENAI_CHAT_CAPABILITIES;
  }
  if (providerType === "anthropic" && isClaude47SamplingRemoved(modelId)) {
    return { ...base, temperature: false, topP: false, topK: false };
  }
  if (providerType === "deepseek" && isDeepSeekReasoningModelId(modelId)) {
    return { ...base, temperature: false, topP: false };
  }
  return base;
}

const DEFAULT_EFFORT_LEVELS = ["low", "medium", "high"] as const;
// OpenRouter ids with no non-reasoning mode. (google/gemini-pro-latest
// was dropped — gateway 404s; don't re-pin to a versioned id that
// may rotate again.)
const OPENROUTER_MANDATORY_REASONING_MODELS = new Set([
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

// Longest prefixes first (find() must match before the bare-family
// fallback). Levels per platform.claude.com/docs/en/about-claude/models/overview.
// 4.5 line maps to budget_tokens; sonnet-4/opus-4 retire 2026-06-15.
const ANTHROPIC_REASONING_MODELS = [
  {
    prefixes: ["claude-opus-4-7"],
    levels: ["none", "low", "medium", "high", "xhigh", "max"],
  },
  {
    prefixes: ["claude-opus-4-6", "claude-sonnet-4-6"],
    levels: ["none", "low", "medium", "high", "max"],
  },
  {
    prefixes: ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
    levels: ["none", "low", "medium", "high"],
  },
  {
    prefixes: ["claude-opus-4-1", "claude-opus-4", "claude-sonnet-4"],
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
      supportsReasoningOff: true,
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
    prefixes: ["gpt-5.5", "gpt-5.4"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "xhigh"],
  },
  {
    prefixes: ["gpt-5.3-chat-latest"],
    supportsOff: false,
    levels: ["medium"],
  },
  {
    // gpt-5.3-codex enum is low/medium/high/xhigh only per dev page.
    prefixes: ["gpt-5.3-codex"],
    supportsOff: false,
    levels: ["low", "medium", "high", "xhigh"],
  },
  {
    // Azure footnote ^7^: minimal supported only on original gpt-5.
    // Listed before the bare gpt-5 entry so the longer match wins.
    prefixes: ["gpt-5.1", "gpt-5.2"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "xhigh"],
  },
  {
    prefixes: ["gpt-5"],
    supportsOff: false,
    levels: ["minimal", "low", "medium", "high"],
  },
  {
    // o-series all accept low/medium/high per dev pages + Azure table.
    prefixes: ["o1", "o3", "o4", "codex-mini"],
    supportsOff: false,
    levels: DEFAULT_EFFORT_LEVELS,
  },
] as const;

function resolveOpenAIReasoningEffortCapabilities(modelId: string): ReasoningCaps {
  const normalized = modelId.trim().toLowerCase();
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
  // Kimi exposes a boolean thinking toggle rather than an effort scale.
  //   - kimi-k2.6:        thinking enabled by default, toggleable
  //                       via extra_body: {thinking: {type: enabled|disabled}}
  //   - kimi-k2-thinking: thinking always on, no off switch
  //   - kimi-k2.5 (and anything else): no thinking
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

// Gemini's thinking ladder.
//   - Gemini 3.x (3 / 3.1 / 3.5, Pro + Flash + Flash-Lite) and the
//     gemini-pro-latest / gemini-flash-latest aliases use the new
//     `thinkingConfig.thinkingLevel` string field (LOW/MEDIUM/HIGH/
//     MINIMAL). Pro tier rejects MINIMAL.
//   - Gemini 2.5 Flash + 2.5 Pro stay on the integer
//     `thinkingConfig.thinkingBudget` (0=off on Flash, -1=dynamic,
//     N>0=cap; Pro rejects 0).
//   - 2.5 Flash-Lite: no native thinking surfaced; leave it off.
//   - Image-tier ids (`*-image*`, `nano-banana-pro-preview`): image
//     generation path -- no reasoning controls.
const GEMINI3_PRO_PREFIXES = [
  "gemini-3.5-pro",
  "gemini-3.1-pro",
  "gemini-3-pro-preview",
  "gemini-pro-latest",
];
const GEMINI3_FLASH_PREFIXES = [
  "gemini-3.5-flash",
  "gemini-3.1-flash",
  "gemini-3-flash",
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
  // Gemini 2.5 Flash-Lite supports `thinkingBudget` with `0` = off and
  // a positive range starting at 512 (the backend maps "minimal" to
  // that floor at external_provider._stream_gemini). Check this branch
  // BEFORE the broader `gemini-2.5-flash` prefix.
  // https://ai.google.dev/gemini-api/docs/thinking
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
  if (GEMINI3_PRO_PREFIXES.some((p) => m.startsWith(p))) {
    // Gemini 3.x Pro: thinkingLevel supports low/medium/high per
    // https://ai.google.dev/gemini-api/docs/thinking and
    // https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-1-pro.
    // Cannot fully disable thinking; "minimal" is rejected on Pro.
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["low", "medium", "high"] as const,
    });
  }
  if (GEMINI3_FLASH_PREFIXES.some((p) => m.startsWith(p))) {
    // Gemini 3 Flash: thinkingLevel minimal/low/medium/high. Minimal
    // is the closest to "off" Google offers on Gemini 3.
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
    // Gemini 2.5 Pro: thinkingBudget cannot be 0 (API rejects with
    // "only works in thinking mode"); backend coerces to a small
    // positive budget. The picker still hides the off switch.
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
  // magistral-*: native always-on (422 on reasoning_effort).
  // mistral-{small,medium,vibe-cli}-latest: none/low/medium/high.
  // https://docs.mistral.ai/studio-api/conversations/reasoning
  if (
    modelId === "magistral-medium-latest" ||
    modelId === "magistral-small-latest"
  ) {
    return withEnableThinkingStyle({
      supportsReasoning: true,
      reasoningAlwaysOn: true,
    });
  }
  if (
    modelId === "mistral-small-latest" ||
    modelId === "mistral-medium-latest" ||
    modelId === "mistral-vibe-cli-latest"
  ) {
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: ["none", "low", "medium", "high"] as const,
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

// vLLM has no per-model reasoning signal on OpenAI-compat — pin via user toggle.
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

// Provider-specific matching lives in the per-provider resolvers
// (resolveOpenAI / Anthropic / Kimi / Mistral...). Unknown providers
// default to no reasoning controls.
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

  // Some OpenRouter-routed ids are mandatory-reasoning and must stay on even
  // if they arrive through aliased/custom provider routes.
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

  const isOpenAIProvider = normalizedProvider === "openai";
  const isAnthropicProvider = normalizedProvider === "anthropic";
  const isKimiProvider = normalizedProvider === "kimi";
  const isMistralProvider = normalizedProvider === "mistral";
  const isOpenRouterProvider = normalizedProvider === "openrouter";
  if (isOpenRouterProvider) {
    // OpenRouter's unified `reasoning` parameter is accepted on every
    // request; gateway no-ops for non-reasoning models. Mandatory ids
    // already handled above; everything else exposes a toggle.
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
    // Custom Gemini OAI-compat gateways (LiteLLM, proxies) route
    // through /chat/completions which drops the Gemini-native
    // thinkingConfig payload. Hide the native thinking ladder so the
    // UI does not advertise a control the backend cannot honor.
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
