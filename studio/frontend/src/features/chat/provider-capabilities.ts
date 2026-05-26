// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Per-provider sampling parameter capability matrix.
 *
 * Values are derived from each provider's published chat-completion docs as of
 * 2026-05. They describe which of our UI knobs map cleanly onto the provider's
 * request body; the panel hides params a provider does not accept so users
 * cannot dial a value that gets silently dropped or rejected.
 *
 * "Local" models (anything that is not an external provider) are represented by
 * a null capability — every knob renders for them.
 */

// NB: when adding a new sampling knob, default it to `false` on every
// SaaS provider in PROVIDER_CAPABILITIES below (only local backends
// + the permissive {custom, vllm, ollama, llama_cpp, openrouter}
// providers should expose llama.cpp-specific samplers).
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
  /**
   * OpenAI-style frequency penalty. Accepted by Chat Completions only.
   * Anthropic and the OpenAI Responses family both reject it (the latter
   * with `Unsupported parameter`).
   */
  frequencyPenalty: boolean;
  /**
   * Best-effort determinism seed. Accepted by OpenAI Chat Completions and
   * most OpenAI-compatible local backends (vLLM, llama.cpp). Rejected by
   * the Responses family and silently dropped by Anthropic.
   */
  seed: boolean;
  /**
   * Custom stop sequences. Maps to `stop` (OpenAI Chat) or `stop_sequences`
   * (Anthropic). Not accepted by the Responses family.
   */
  stop: boolean;
  /**
   * Provider service tier (`auto` / `standard_only` for Anthropic,
   * `auto`/`default`/`flex`/`priority`(+`scale`) for OpenAI). See
   * {@link getServiceTierOptions} for the legal values per provider.
   */
  serviceTier: boolean;
  /**
   * Whether the provider supports turning off parallel tool dispatch.
   * Maps to `parallel_tool_calls: false` on both OpenAI APIs and
   * `disable_parallel_tool_use: true` on Anthropic (inverted).
   */
  parallelToolCalls: boolean;
  /**
   * llama.cpp `typ_p` (locally typical sampling). Local llama-server
   * only — no SaaS provider currently accepts this field. Default is
   * `false` for every external provider and `true` only for the local
   * permissive {custom, vllm, ollama, llama_cpp} buckets.
   */
  typicalP: boolean;
  /**
   * llama.cpp `top_n_sigma` sampler (newer top-sigma cutoff). Local
   * only; -1 disables.
   * https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
   */
  topNSigma: boolean;
  /**
   * llama.cpp repetition window (`repeat_last_n`). Pairs with
   * `repeat_penalty`. Local only; 0 disables, -1 = ctx-size.
   */
  repeatLastN: boolean;
  /**
   * llama.cpp dynamic temperature range (`dynatemp_range`). Local
   * only; 0.0 disables.
   */
  dynatempRange: boolean;
  /**
   * llama.cpp dynamic temperature exponent (`dynatemp_exponent`).
   * Local only. Paired with dynatempRange.
   */
  dynatempExponent: boolean;
  /**
   * llama.cpp Mirostat sampling mode (`mirostat`). Local only.
   * 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0.
   */
  mirostat: boolean;
  /**
   * llama.cpp Mirostat target entropy (`mirostat_tau`). Local only.
   * Only meaningful when mirostat != 0.
   */
  mirostatTau: boolean;
  /**
   * llama.cpp Mirostat learning rate (`mirostat_eta`). Local only.
   * Only meaningful when mirostat != 0.
   */
  mirostatEta: boolean;
  /**
   * OpenRouter `top_a` (alternate dynamic-top-P). Documented at
   * https://openrouter.ai/docs/api/reference/parameters. Other
   * gateways silently drop it; we surface it only for openrouter.
   */
  topA: boolean;
}

/**
 * Per-provider stop-sequence max count. Resolved by
 * `getProviderStopMax(providerType)`. Mirrors the backend's
 * `provider_info.stop_max` for the same provider type.
 *   - openai:     4   (Chat Completions hard cap; Responses drops stop)
 *   - anthropic:  16  (client-side guard; docs publish no max)
 *   - kimi:       5   (https://platform.kimi.ai/docs/api/chat)
 *   - deepseek:   16  (https://api-docs.deepseek.com/api/create-chat-completion)
 *   - mistral:    16  (no documented max; widen to permissive default)
 *   - gemini:     4   (https://ai.google.dev/gemini-api/docs/openai inherits OAI cap)
 *   - openrouter: 4   (normalises to OpenAI's chat schema)
 *   - default:    16  (covers ollama, vllm, llama.cpp, custom)
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
 * Legal `service_tier` values per provider. Anthropic exposes only
 * `auto` and `standard_only`. OpenAI in Studio is routed through
 * `/v1/responses`, which the live docs list as
 * `auto|default|flex|priority`; `scale` is excluded here even though
 * the openai-python SDK type happens to include it. Other providers
 * fall through to a permissive `auto` / `default` pair so the picker
 * stays usable for OpenAI-compat backends.
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
 * Output-token cap for any external provider request. Picked to stay below the
 * tightest declared limit across the providers we ship (Anthropic Claude Opus
 * tops out at 128k, GPT-5.x ~128k, Gemini 2.5 ~65k, DeepSeek 8k) while staying
 * well above what a typical chat reply needs. The local-model path is not
 * subject to this — local backends honour whatever the loaded context allows.
 *
 * If a user's stored maxTokens (e.g. carried over from a prior local-model
 * session with a 128k+ context) exceeds this, chat-adapter clamps the
 * outbound request so the provider does not 400 on it.
 */
export const EXTERNAL_MAX_OUTPUT_TOKENS = 32768;

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
): boolean {
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
  // Family boundary ("" or "-") required so IDs like "claude-opus-4-70"
  // / "claude-opus-4-7b" do not match.
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
 * Returns false for every other provider. The backend additionally
 * gates the OpenAI shell tool on `is_openai_cloud` so custom
 * OpenAI-compat servers (ollama / llama.cpp / vLLM) that also report
 * `provider_type="openai"` never receive the tool — but in practice
 * none of those catalogs surface the `gpt-5.5` ids anyway, so the
 * frontend prefix match is enough.
 *
 * v1 wires the tools themselves; file uploads (Anthropic
 * `container_upload` / OpenAI `input_file`) are a deliberate follow-up.
 */
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
 * managed cloud (api.openai.com), as opposed to a custom OpenAI-compat
 * backend (ollama / llama.cpp / vLLM / generic "custom" preset). The
 * shell tool ONLY exists on OpenAI cloud; sending it to anything else
 * 400s the request. Mirror of the backend's
 * `is_openai_cloud = "api.openai.com" in self.base_url` guard.
 */
function isOpenAICloudBaseUrl(baseUrl: string | null | undefined): boolean {
  if (!baseUrl) return true; // No override → uses the default openai.com base.
  return baseUrl.trim().toLowerCase().includes("api.openai.com");
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
  return false;
}

/**
 * Whether the selected external provider/model exposes OpenAI's
 * Responses-API server-side image_generation tool. Lit on for OpenAI
 * cloud (`api.openai.com`) when the picked model is a Responses-API
 * family id (gpt-5.x today). The backend additionally gates on
 * `is_openai_cloud`; mirror that here so the pill is hidden on custom
 * OpenAI-compat backends (ollama / llama.cpp / vLLM) that report
 * `provider_type="openai"` but would 400 on a `{type:"image_generation"}`
 * tool. See backend/core/inference/external_provider.py near line 2770
 * for the dispatch and backend/tests/test_openai_image_generation.py
 * for the round-trip coverage.
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
  if (providerType !== "openai") return false;
  if (!isOpenAICloudBaseUrl(baseUrl)) return false;
  const normalized = modelId?.trim().toLowerCase() ?? "";
  if (!normalized) return false;
  return OPENAI_IMAGE_GENERATION_MODEL_PREFIXES.some((prefix) =>
    normalized.startsWith(prefix),
  );
}

/**
 * Per-provider minimum on the outbound max_tokens. Kimi's docs require
 * `max_tokens >= 16000` whenever a thinking model is in use so the
 * reasoning_content and final answer both fit in the budget — anything
 * lower truncates the response mid-stream. Other providers don't have a
 * documented floor, so they fall through to the generic min of 64 in
 * the slider.
 *
 * The chat-adapter resolves the effective floor on send and bumps the
 * outbound max_tokens up to this value if the user's stored maxTokens
 * sits below it. The settings panel reflects the same floor as the
 * slider min so the displayed value never drifts from what's sent.
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
};

// Local llama.cpp-style backends (own llama-server, vLLM with extended
// sampler support, Ollama). Exposes the full llama.cpp sampler chain
// (typical_p / top_n_sigma / mirostat / dynatemp / repeat_last_n) but
// not OpenRouter's gateway-specific top_a.
const LOCAL_LLAMA_CAPABILITIES: ProviderCapabilities = {
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
};

// OpenRouter is a router-of-routers: the gateway accepts a wider set
// of OpenAI-style sampling fields than any single upstream supports
// and silently drops what the chosen route does not, per
// https://openrouter.ai/docs/api/reference/parameters. Surface the
// router's full documented set (incl. top_a) and leave the
// llama.cpp-only knobs off (the docs don't list them, so we don't
// either even though many openrouter routes terminate at llama.cpp).
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
};

// Reasoning-class OpenAI models served via /v1/responses fix temperature
// at 1, ignore top_p, and 400 on presence/frequency_penalty / seed. Non
// reasoning models (gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo) keep the
// full sampling surface even when routed through /v1/responses. See
// https://platform.openai.com/docs/guides/reasoning and the GPT-5 release
// notes; backend dispatch is external_provider._stream_openai_responses.
// The Responses API itself drops `stop`, so we leave that off for all
// OpenAI models regardless of family.
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

// Mirror of backend _ANTHROPIC_4_7_SAMPLING_REMOVED in
// studio/backend/core/inference/external_provider.py:110. Claude 4.7
// (Opus/Sonnet/Haiku) removed temperature, top_p, and top_k entirely;
// surfacing the sliders would let the user move a control that the
// backend silently strips. The trailing -4-7[-.]/EOL anchor keeps future
// families (claude-opus-5 etc.) unaffected.
const ANTHROPIC_4_7_SAMPLING_REMOVED_REGEX = /^claude-(?:opus|sonnet|haiku)-4-7(?:[-.]|$)/i;

function isClaude47SamplingRemoved(modelId: string | null | undefined): boolean {
  const normalized = modelId?.trim().toLowerCase() ?? "";
  if (!normalized) return false;
  return ANTHROPIC_4_7_SAMPLING_REMOVED_REGEX.test(normalized);
}

// DeepSeek reasoning-class models silently ignore temperature, top_p,
// presence_penalty, frequency_penalty and 400 on logprobs/top_logprobs.
// `deepseek-reasoner` is the dedicated thinking model;
// `deepseek-v4-flash` runs reasoning-mode under the same flag as well.
// Match by prefix so future revisions (deepseek-reasoner-2027 etc.)
// continue to gate correctly.
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
  // Default OpenAI bucket is reasoning-class (current registry only ships
  // gpt-5.x / o3 ids), but per-model resolution in getProviderCapabilities
  // upgrades non-reasoning ids (gpt-4o etc.) to OPENAI_CHAT_CAPABILITIES.
  openai: OPENAI_REASONING_CAPABILITIES,
  // Anthropic's Messages API accepts top_k on 3.x and 4.5/4.6, but Claude
  // 4.7 (Opus/Sonnet/Haiku) deprecated it and returns 400 if it is set.
  // We surface top_k in the panel for all Anthropic providers and let the
  // backend strip it per-model — see _stream_anthropic in
  // studio/backend/core/inference/external_provider.py.
  // Presence/frequency penalty / seed / logprobs are not part of the
  // Messages API on any Claude generation. stop_sequences (Anthropic name
  // for `stop`), service_tier (auto|standard_only), and
  // disable_parallel_tool_use (inverse of parallel_tool_calls) ARE
  // supported.
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
  },
  mistral: OPENAI_COMPAT_BASE,
  gemini: OPENAI_COMPAT_BASE,
  // Kimi k2.5/k2.6 are reasoning-class; the API locks temperature
  // and top_p to fixed defaults and 400s on any other value:
  //   "invalid temperature: only 1 is allowed for this model".
  // Hide both sliders so the user is not offered knobs the model
  // silently overrides. Backend additionally strips these fields via
  // PROVIDER_REGISTRY['kimi']['body_omit']. seed and parallel_tool_
  // calls are not in Kimi's documented Chat Completion schema
  // (https://platform.kimi.ai/docs/api/chat); hide them so users are
  // not offered controls that the upstream may silently drop or 400.
  kimi: {
    temperature: false,
    topP: false,
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: true,
    // K2.5/K2.6 lock sampling the same way temperature/top_p are
    // locked; reviewers report non-default frequency_penalty 400s
    // upstream, so hide the slider and strip the field in body_omit.
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
  },
  // DeepSeek deprecated presence/frequency penalty in their current docs.
  // Chat-class defaults (deepseek-chat / deepseek-v4-flash non-thinking):
  // accept temperature, top_p, seed, stop. Reasoning class
  // (deepseek-reasoner / deepseek-v4-flash thinking-mode) ignores
  // temperature, top_p, presence_penalty, frequency_penalty entirely and
  // 400s on logprobs — see
  // https://api-docs.deepseek.com/guides/reasoning_model. Per-model
  // resolution in getProviderCapabilities downshifts reasoner ids onto
  // DEEPSEEK_REASONING_CAPABILITIES.
  deepseek: {
    temperature: true,
    topP: true,
    topK: false,
    minP: false,
    repetitionPenalty: false,
    presencePenalty: false,
    frequencyPenalty: false,
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
  },
  qwen: OPENAI_COMPAT_BASE,
  huggingface: OPENAI_COMPAT_BASE,
  // OpenRouter surfaces the gateway's documented sampling field set
  // (incl. top_a). llama.cpp-specific knobs (typical_p, mirostat,
  // dynatemp, top_n_sigma, repeat_last_n) are gated off because the
  // OpenRouter API docs do not list them; they would be silently
  // dropped on most underlying models.
  openrouter: OPENROUTER_CAPABILITIES,
  // Local OpenAI-compatible connections terminate at llama-server-
  // style backends — full llama.cpp sampler chain available.
  custom: LOCAL_LLAMA_CAPABILITIES,
  vllm: LOCAL_LLAMA_CAPABILITIES,
  ollama: LOCAL_LLAMA_CAPABILITIES,
  llama_cpp: LOCAL_LLAMA_CAPABILITIES,
};

const DEFAULT_EXTERNAL_CAPABILITIES = OPENAI_COMPAT_BASE;

/**
 * Resolve the capability set for an external provider, optionally
 * specialised by model id. Returns `null` for a local model (i.e. when
 * `providerType` is null/undefined), which callers should treat as
 * "every knob applies".
 *
 * Per-model specialisations:
 *   - openai + non-reasoning model (gpt-4o, gpt-4-turbo, gpt-4,
 *     gpt-3.5-turbo): full sampling surface (OPENAI_CHAT_CAPABILITIES).
 *   - openai + reasoning model (gpt-5.x, o1, o3, o4): restrictive
 *     (OPENAI_REASONING_CAPABILITIES).
 *   - anthropic + claude-*-4-7: temperature/top_p/top_k stripped to
 *     match the backend 400-avoidance regex.
 *   - deepseek + reasoning model (deepseek-reasoner / r1): hides
 *     temperature/top_p (silently ignored upstream).
 */
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
 * resolve external-model thinking capabilities.
 * provider-specific matching lives in the OpenAI/Anthropic resolvers.
 * other providers default to no reasoning controls.
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
    // OpenRouter's unified `reasoning` parameter is accepted on every
    // chat-completion request; the gateway silently no-ops for models
    // that don't reason. Mandatory-reasoning ids are handled by the
    // early guard above; everything else exposes a toggleable control.
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
