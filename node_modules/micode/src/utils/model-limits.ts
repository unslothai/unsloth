// Shared model context limits (tokens)
// Used by context-window-monitor and auto-compact hooks

// Fallback patterns for models not in opencode.json
export const MODEL_CONTEXT_LIMITS: Record<string, number> = {
  // Claude models
  "claude-opus": 200_000,
  "claude-sonnet": 200_000,
  "claude-haiku": 200_000,
  "claude-3": 200_000,
  "claude-4": 200_000,
  // OpenAI models
  "gpt-4o": 128_000,
  "gpt-4-turbo": 128_000,
  "gpt-4": 128_000,
  "gpt-5": 200_000,
  o1: 200_000,
  o3: 200_000,
  // Google models
  gemini: 1_000_000,
};

export const DEFAULT_CONTEXT_LIMIT = 200_000;

/**
 * Get the context window limit for a given model.
 * Priority: loaded limits (exact match) > pattern match > default
 *
 * @param modelID - The model ID (e.g., "gpt-4o", "claude-opus")
 * @param providerID - Optional provider ID (e.g., "openai", "anthropic")
 * @param loadedLimits - Optional map of "provider/model" -> limit from opencode.json
 */
export function getContextLimit(modelID: string, providerID?: string, loadedLimits?: Map<string, number>): number {
  // Check loaded limits first (exact match with provider/model)
  if (loadedLimits && providerID) {
    const exactKey = `${providerID}/${modelID}`;
    const exactLimit = loadedLimits.get(exactKey);
    if (exactLimit !== undefined) {
      return exactLimit;
    }
  }

  // Fall back to pattern matching on model ID
  const modelLower = modelID.toLowerCase();
  for (const [pattern, limit] of Object.entries(MODEL_CONTEXT_LIMITS)) {
    if (modelLower.includes(pattern)) {
      return limit;
    }
  }

  return DEFAULT_CONTEXT_LIMIT;
}
