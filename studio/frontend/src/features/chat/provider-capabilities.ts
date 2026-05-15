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
  gemini: OPENAI_COMPAT_BASE,
  // Kimi k2.5/k2.6 are reasoning-class — the API locks temperature and
  // top_p to fixed defaults and 400s on any other value:
  //   "invalid temperature: only 1 is allowed for this model".
  // Hide both sliders so the user is not offered knobs the model
  // silently overrides. Backend additionally strips these fields via
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
  // Custom providers are assumed OpenAI-compatible by the backend; users who
  // point at vLLM/Ollama backends often want top_k / min_p / repetition,
  // so be permissive.
  custom: ALL_SUPPORTED,
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

/**
 * resolve external-model thinking capabilities.
 * provider-specific matching lives in the OpenAI/Anthropic resolvers.
 * other providers default to no reasoning controls.
 */
export function getExternalReasoningCapabilities(
  providerType: string | null | undefined,
  modelId: string | null | undefined,
): ExternalReasoningCapabilities {
  const normalizedModel = modelId?.trim().toLowerCase() ?? "";
  const normalizedProvider = providerType?.trim().toLowerCase() ?? "";
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
