// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Per-provider sampling parameter capability matrix.
 *
 * Derived from each provider's published chat-completion docs (2026-05). The
 * panel hides params a provider does not accept so users cannot dial a value
 * that gets silently dropped or rejected. "Local" models (non-external) use a
 * null capability — every knob renders for them.
 */

export interface ProviderCapabilities {
  /**
   * Temperature sampling. Reasoning-class models (OpenAI's gpt-5.x / o3 via
   * /v1/responses) reject this with `Unsupported parameter`.
   */
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
  // Mirrors the store's ReasoningStyle. External providers only ever use the
  // first two; "enable_thinking_effort" exists so a local model's caps can be
  // assigned here without narrowing.
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

/**
 * Pick a stored reasoning effort level that exists in `effortLevels`, mapping
 * legacy "xhigh" to "max" when only the latter is exposed (Claude 4.6).
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

/**
 * Per-model max-output caps from each provider's docs:
 *   OpenAI:    developers.openai.com/api/docs/models/gpt-5.5
 *   Anthropic: platform.claude.com/docs/en/about-claude/models
 *   Gemini:    ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview
 *   DeepSeek:  api-docs.deepseek.com/quick_start/pricing (V4 family)
 * Local-model path is unaffected.
 */
const EXTERNAL_MAX_OUTPUT_TOKENS_BY_MODEL: Array<{
  providerType: string;
  prefixes: readonly string[];
  cap: number;
}> = [
  // OpenAI
  { providerType: "openai", prefixes: ["gpt-5.5-pro", "gpt-5.5"], cap: 128000 },
  { providerType: "openai", prefixes: ["gpt-5.4-pro", "gpt-5.4"], cap: 65536 },
  { providerType: "openai", prefixes: ["gpt-5.3"], cap: 16384 },
  // Anthropic
  {
    providerType: "anthropic",
    prefixes: ["claude-opus-4-7"],
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
    ],
    cap: 64000,
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

/**
 * Documented per-model output cap; unknown ids fall back to
 * `EXTERNAL_MAX_OUTPUT_TOKENS` (32k). OpenRouter `provider/model` ids have the
 * prefix stripped before matching.
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

/**
 * Whether the external provider offers a built-in web-search tool that the
 * model invokes server-side. When `true`, the chat composer's Search button
 * is available for that provider and the chat-adapter forwards
 * `enable_tools: true, enabled_tools: ["web_search"]` on the request — the
 * backend routes the call through the provider's tool schema:
 *   - OpenAI:     `tools: [{type: "web_search"}]` on /v1/responses
 *   - Anthropic:  `tools: [{type: "web_search_20250305", name: "web_search",
 *                           max_uses: 5}]` on /v1/messages
 *   - OpenRouter: `plugins: [{id: "web"}]` on /v1/chat/completions (the
 *                 router's universal web-search shape; works for every
 *                 underlying model including the `openrouter/free` router).
 *   - Kimi:       `tools: [{type: "builtin_function", function: {name:
 *                          "$web_search"}}]` with `thinking: {type:
 *                          "disabled"}`. Requires a client round-trip:
 *                 the first call returns the search args; the backend
 *                 echoes them back as a role=tool message; the second
 *                 call streams the answer. Handled in
 *                 _stream_kimi_web_search on the backend.
 *
 * Mistral is intentionally excluded: their `web_search` connector lives on
 * the Agents API (`/v1/agents` + `/v1/conversations`), not chat completions,
 * and returns `"WebSearchTool connector is not supported"` if injected into
 * /v1/chat/completions. Wiring it would require a dedicated Agents streaming
 * path. Gemini's grounded-search can be added with the same pattern when
 * matching backend translation lands.
 */
export function providerSupportsBuiltinWebSearch(
  providerType: string | null | undefined,
  modelId?: string | null | undefined,
  baseUrl?: string | null | undefined,
): boolean {
  // Gemini ships grounded search via `tools: [{googleSearch: {}}]` on every
  // chat-capable model. Most image-tier ids reject text-tool wiring (the
  // responseModalities path excludes text tools), but Google documents Search
  // grounding on the Gemini 3 image family (gemini-3-pro-image-preview,
  // gemini-3.1-flash-image-preview, nano-banana-pro), so allow it there and
  // hide on older
  // image ids. Custom Gemini OpenAI-compat proxies skip the backend's native
  // translator, so native tool envelopes never reach them -- hide the pill.
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

/**
 * Whether the external provider exposes a server-side web_fetch tool
 * (single URL, text or PDF) emitting a document block. Anthropic-only
 * today (`web_fetch_20250910` / `web_fetch_20260209`). Gates the
 * composer's standalone Fetch pill, independent of Search.
 */
export function providerSupportsBuiltinWebFetch(
  providerType: string | null | undefined,
): boolean {
  return providerType === "anthropic";
}

/**
 * Whether the active provider + model supports Anthropic fast-mode
 * (`speed: "fast"` + `fast-mode-2026-02-01` header). Opus 4.6 / 4.7
 * only per https://platform.claude.com/docs/en/build-with-claude/fast-mode.
 * Backend silently drops on unsupported models as a second defence.
 */
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
  // Family boundary ("" or "-") required so IDs like "claude-opus-4-70" or
  // "claude-opus-4-7b" do not match.
  return ANTHROPIC_FAST_MODE_MODEL_PREFIXES.some(
    (prefix) => modelId === prefix || modelId.startsWith(`${prefix}-`),
  );
}

/**
 * Whether the selected external provider/model exposes a server-side
 * code-execution tool. Two providers ship one today:
 *
 *   - **Anthropic** (`code_execution_20250825`): Python + bash +
 *     str_replace-based file edits inside a 5 GB sandboxed container
 *     per request. Documented at
 *       https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
 *
 *   - **OpenAI cloud** (`shell` on /v1/responses): bash inside a
 *     reusable container; we auto-create one on the first turn of a
 *     chat thread and reference it on subsequent turns via the
 *     thread's stored `openaiCodeExecContainerId`. Documented at
 *       https://developers.openai.com/api/docs/guides/tools-shell
 *
 * Returns false for every other provider. The backend also gates the OpenAI
 * shell tool on `is_openai_cloud` so custom OpenAI-compat servers reporting
 * `provider_type="openai"` never receive it; in practice those catalogs don't
 * surface the `gpt-5.5` ids, so the frontend prefix match suffices.
 *
 * v1 wires the tools only; file uploads (Anthropic `container_upload` / OpenAI
 * `input_file`) are a deliberate follow-up.
 */
const ANTHROPIC_CODE_EXECUTION_MODEL_PREFIXES = [
  "claude-opus-4-7",
  "claude-opus-4-6",
  "claude-sonnet-4-6",
  "claude-opus-4-5",
  "claude-sonnet-4-5",
  "claude-haiku-4-5",
  // Deprecated upstream but the registry still exposes these ids; keep the
  // pill working for users on those snapshots.
  "claude-opus-4-1",
  "claude-opus-4",
  "claude-sonnet-4",
] as const;

// OpenAI cloud shell-tool gating. Docs only show gpt-5.5; gpt-5.5-pro shares
// the same /v1/responses contract. `gpt-5.5-pro` is checked first so the prefix
// match doesn't collide with e.g. a hypothetical `gpt-5.5-turbo`.
const OPENAI_CODE_EXECUTION_MODEL_PREFIXES = [
  "gpt-5.5-pro",
  "gpt-5.5",
] as const;

/**
 * Strict check that a provider config points at OpenAI managed cloud
 * (api.openai.com) or Azure OpenAI Foundry (*.openai.azure.com), not a custom
 * OpenAI-compat backend. The shell and image-generation tools exist only on
 * cloud backends; sending them elsewhere 400s. Mirrors the backend's
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

/**
 * Whether the selected provider/model exposes OpenAI's Responses-API
 * image_generation tool. On for OpenAI cloud (`api.openai.com`) with a
 * Responses-API family id (gpt-5.x). Mirrors the backend's `is_openai_cloud`
 * gate so the pill hides on custom OpenAI-compat backends reporting
 * `provider_type="openai"` that would 400 on a `{type:"image_generation"}` tool.
 * See backend/core/inference/external_provider.py (~line 2770) for dispatch and
 * backend/tests/test_openai_image_generation.py for round-trip coverage.
 */
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
    // Gemini Nano Banana image-output ids carry `-image` or the `nano-banana`
    // alias. The backend flips responseModalities to ["TEXT", "IMAGE"] and maps
    // inlineData parts into the same image_b64 tool_end envelope as the OpenAI
    // path so the UI renders inline. Custom Gemini OpenAI-compat proxies skip
    // the native translator, so hide the image pill there.
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
 * Whether the saved Gemini connection points at a custom OpenAI-compat gateway
 * (any non-Google host). The backend `_is_openai_compatible` routes these
 * through `/chat/completions` instead of the native translator, so native Gemini
 * tool envelopes never reach them. Hide the matching Unsloth pills here so the
 * request, builder, and UI agree.
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

/**
 * Per-provider minimum on the outbound max_tokens. Kimi requires
 * `max_tokens >= 16000` for thinking models so reasoning_content and the answer
 * both fit; lower truncates mid-stream. Others fall through to the generic 64.
 *
 * The chat-adapter resolves the floor on send and bumps maxTokens up to it if
 * the stored value is below; the settings slider min reflects the same floor so
 * the displayed value never drifts from what's sent.
 */
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
  // OpenAI's flagship models (gpt-5.x / o3 / gpt-4.5) are reasoning-class
  // models served via /v1/responses, which rejects temperature, top_p, and
  // presence/frequency penalty. See backend
  // external_provider._stream_openai_responses for the proxy.
  openai: {
    temperature: false,
    topP: false,
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
  },
  // Anthropic's Messages API accepts top_k on 3.x and 4.5/4.6, but Claude
  // 4.7 (Opus/Sonnet/Haiku) deprecated it and returns 400 if it is set.
  // We surface top_k in the panel for all Anthropic providers and let the
  // backend strip it per-model — see _stream_anthropic in
  // studio/backend/core/inference/external_provider.py.
  // Presence/frequency penalty is not part of the Messages API on any
  // Claude generation.
  anthropic: {
    temperature: true,
    topP: true,
    topK: true,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
  },
  mistral: OPENAI_COMPAT_BASE,
  // Gemini's generationConfig accepts temperature, topP, topK and
  // presencePenalty (plus a frequencyPenalty we don't surface). minP and
  // repetitionPenalty aren't part of the contract --
  // see https://ai.google.dev/api/rest/v1beta/GenerationConfig. Backend shaping
  // lives in _stream_gemini in
  // studio/backend/core/inference/external_provider.py.
  gemini: {
    temperature: true,
    topP: true,
    topK: true,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: true,
  },
  // Kimi k2.5/k2.6 are reasoning-class — the API locks temperature and top_p to
  // fixed defaults and 400s on any other value ("invalid temperature: only 1 is
  // allowed for this model"). Hide both sliders. Backend also strips these via
  // PROVIDER_REGISTRY['kimi']['body_omit'].
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
  // OpenRouter silently drops params the target model does not support, so we
  // surface every knob and let the gateway handle the per-model fan-out.
  openrouter: ALL_SUPPORTED,
  // Local OpenAI-compat connections go through the OpenAI backend path, but
  // vLLM/Ollama/llama.cpp users often want top_k/min_p/repetition controls, so
  // be permissive.
  custom: ALL_SUPPORTED,
  vllm: ALL_SUPPORTED,
  ollama: ALL_SUPPORTED,
  llama_cpp: ALL_SUPPORTED,
};

const DEFAULT_EXTERNAL_CAPABILITIES = OPENAI_COMPAT_BASE;

/**
 * Resolve the capability set for an external provider. Returns `null` for
 * a local model (i.e. when `providerType` is null/undefined), which callers
 * should treat as "every knob applies".
 */
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
    prefixes: ["claude-opus-4-7"],
    levels: ["none", "low", "medium", "high", "xhigh", "max"],
  },
  {
    prefixes: ["claude-opus-4-6", "claude-sonnet-4-6"],
    levels: ["none", "low", "medium", "high", "max"],
  },
  {
    prefixes: ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
    // Backend maps semantic levels to manual budget_tokens.
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
    prefixes: ["gpt-5.3-codex"],
    supportsOff: true,
    levels: ["none", "low", "medium", "high", "xhigh"],
  },
  {
    prefixes: ["gpt-5", "gpt-5.1", "gpt-5.2"],
    supportsOff: false,
    levels: ["minimal", "low", "medium", "high"],
  },
  {
    prefixes: ["o3"],
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
  // Kimi exposes a boolean thinking toggle, not an effort scale.
  //   - kimi-k2.6:        on by default, toggleable via
  //                       extra_body: {thinking: {type: enabled|disabled}}
  //   - kimi-k2-thinking: always on, no off switch
  //   - kimi-k2.5 (and others): no thinking
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
//   - Gemini 3.x (Pro + Flash + Flash-Lite) and the gemini-pro-latest /
//     gemini-flash-latest aliases use the string `thinkingConfig.thinkingLevel`
//     (LOW/MEDIUM/HIGH/MINIMAL); Pro tier rejects MINIMAL.
//   - Gemini 2.5 Flash + 2.5 Pro use the integer `thinkingConfig.thinkingBudget`
//     (0=off on Flash, -1=dynamic, N>0=cap; Pro rejects 0).
//   - 2.5 Flash-Lite: no native thinking surfaced; leave off.
//   - Image-tier ids: image generation path -- no reasoning controls.
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
  // Gemini 2.5 Flash-Lite: `thinkingBudget` 0 = off, positive range from 512
  // (backend maps "minimal" to that floor in _stream_gemini). Check this branch
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
    // Gemini 3.x Pro: thinkingLevel low/medium/high; cannot fully disable, and
    // "minimal" is rejected on Pro. Refs:
    // https://ai.google.dev/gemini-api/docs/thinking and
    // https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-1-pro.
    return withReasoningEffortStyle({
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["low", "medium", "high"] as const,
    });
  }
  if (GEMINI3_FLASH_PREFIXES.some((p) => m.startsWith(p))) {
    // Gemini 3 Flash: thinkingLevel minimal/low/medium/high. Minimal is the
    // closest to "off" Google offers on Gemini 3.
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
    // Gemini 2.5 Pro: thinkingBudget cannot be 0 (API rejects "only works in
    // thinking mode"); backend coerces to a small positive budget. Hide the off
    // switch in the picker.
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

/**
 * Resolve external-model thinking capabilities. Provider-specific matching lives
 * in the per-provider resolvers; others default to no reasoning controls.
 */
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
    // OpenRouter's unified `reasoning` param is accepted on every request; the
    // gateway no-ops for non-reasoning models. Mandatory-reasoning ids are
    // handled by the early guard; everything else gets a toggleable control.
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
    // Custom Gemini OAI-compat gateways route through /chat/completions, which
    // drops the native thinkingConfig payload. Hide the native thinking ladder
    // so the UI doesn't advertise a control the backend can't honor.
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
