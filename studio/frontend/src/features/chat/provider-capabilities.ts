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
    | "xhigh"
  )[];
};

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

function resolveAnthropicReasoningEffortCapabilities(modelId: string): {
  supportsReasoning: boolean;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: ExternalReasoningCapabilities["reasoningEffortLevels"];
} {
  const normalized = modelId.trim().toLowerCase();
  if (normalized.startsWith("claude-opus-4-7")) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: ["none", "low", "medium", "high", "xhigh"],
    };
  }
  if (
    normalized.startsWith("claude-opus-4-6") ||
    normalized.startsWith("claude-sonnet-4-6")
  ) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: ["none", "low", "medium", "high", "xhigh"],
    };
  }
  // Claude 4.7 and 4.6 use adaptive thinking with effort controls.
  if (
    normalized.startsWith("claude-sonnet-4-7") ||
    normalized.startsWith("claude-haiku-4-7") ||
    normalized.startsWith("claude-haiku-4-6")
  ) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: ["none", "low", "medium", "high"],
    };
  }
  // Claude 4.5 still uses manual thinking budgets; we keep the same semantic
  // UI levels and map them server-side.
  if (
    normalized.startsWith("claude-opus-4-5") ||
    normalized.startsWith("claude-sonnet-4-5") ||
    normalized.startsWith("claude-haiku-4-5")
  ) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: ["none", "low", "medium", "high"],
    };
  }
  return {
    supportsReasoning: false,
    supportsReasoningOff: false,
    reasoningEffortLevels: DEFAULT_EFFORT_LEVELS,
  };
}

function resolveOpenAIReasoningEffortCapabilities(modelId: string): {
  supportsReasoning: boolean;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: ExternalReasoningCapabilities["reasoningEffortLevels"];
} {
  const normalized = modelId.trim().toLowerCase();
  if (
    normalized.startsWith("gpt-5.5-pro") ||
    normalized.startsWith("gpt-5.4-pro")
  ) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["medium", "high", "xhigh"],
    };
  }
  if (normalized.startsWith("gpt-5.5") || normalized.startsWith("gpt-5.4")) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: ["none", "low", "medium", "high", "xhigh"],
    };
  }
  if (normalized.startsWith("gpt-5.3-chat-latest")) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["medium"],
    };
  }
  if (normalized.startsWith("gpt-5.3-codex")) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: true,
      reasoningEffortLevels: ["none", "low", "medium", "high", "xhigh"],
    };
  }
  if (
    normalized === "gpt-5" ||
    normalized.startsWith("gpt-5.1") ||
    normalized.startsWith("gpt-5.2") ||
    normalized.startsWith("gpt-5.3")
  ) {
    return {
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["minimal", "low", "medium", "high"],
    };
  }
  if (normalized.startsWith("o3")) {
    // Keep o3 conservative until OpenAI publishes a per-model effort table.
    return {
      supportsReasoning: true,
      supportsReasoningOff: false,
      reasoningEffortLevels: DEFAULT_EFFORT_LEVELS,
    };
  }
  return {
    supportsReasoning: false,
    supportsReasoningOff: false,
    reasoningEffortLevels: DEFAULT_EFFORT_LEVELS,
  };
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
    return {
      supportsReasoning: false,
      reasoningStyle: "enable_thinking",
      reasoningAlwaysOn: false,
      supportsReasoningOff: false,
      reasoningEffortLevels: DEFAULT_EFFORT_LEVELS,
    };
  }

  // OpenRouter ids are namespaced (e.g. "openai/gpt-5.5").
  const modelForMatching =
    normalizedProvider === "openrouter" && normalizedModel.includes("/")
      ? normalizedModel.split("/").at(-1) ?? normalizedModel
      : normalizedModel;

  const isOpenAIProvider = normalizedProvider === "openai";
  const isAnthropicProvider = normalizedProvider === "anthropic";
  if (!isOpenAIProvider && !isAnthropicProvider) {
    return {
      supportsReasoning: false,
      reasoningStyle: "enable_thinking",
      reasoningAlwaysOn: false,
      supportsReasoningOff: false,
      reasoningEffortLevels: DEFAULT_EFFORT_LEVELS,
    };
  }

  const providerCaps = isOpenAIProvider
    ? resolveOpenAIReasoningEffortCapabilities(modelForMatching)
    : resolveAnthropicReasoningEffortCapabilities(modelForMatching);
  if (providerCaps.supportsReasoning) {
    return {
      supportsReasoning: true,
      reasoningStyle: "reasoning_effort",
      reasoningAlwaysOn: false,
      supportsReasoningOff: providerCaps.supportsReasoningOff,
      reasoningEffortLevels: providerCaps.reasoningEffortLevels,
    };
  }

  return {
    supportsReasoning: false,
    reasoningStyle: "enable_thinking",
    reasoningAlwaysOn: false,
    supportsReasoningOff: false,
    reasoningEffortLevels: DEFAULT_EFFORT_LEVELS,
  };
}
