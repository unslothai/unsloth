


/**
 * Resuming a response that stopped early: Max Tokens ran out (`length`), Stop was
 * pressed (`cancelled`), or the stream was cut (`interrupted`).
 *
 * Continuing re-sends the conversation with the partial as the final assistant turn
 * plus `continue_final_message`, so the prompt ends mid-sentence and the model emits
 * the next token. The new text is appended to the partial.
 */

/** Why a turn ended before the model was done. */
export type IncompleteReason = "length" | "cancelled" | "interrupted";

/** Metadata stamped on an assistant message that stopped early. */
export type IncompleteInfo = {
  reason: IncompleteReason;
};

const INCOMPLETE_REASONS: readonly IncompleteReason[] = [
  "length",
  "cancelled",
  "interrupted",
];

/** Below this a shared boundary is likely coincidence, and trimming would eat output. */
const MIN_OVERLAP = 12;

/** Only the tail can be re-emitted, so the scan stays bounded. */
const MAX_OVERLAP = 400;

/** How much of the partial's opening a restart has to reproduce to be called a restart. */
const RESTART_PROBE = 48;

/** Read the incomplete marker off an assistant message's metadata. */
export function readIncompleteInfo(metadata: unknown): IncompleteInfo | null {
  const custom = (metadata as { custom?: Record<string, unknown> } | undefined)
    ?.custom;
  const incomplete = custom?.incomplete as { reason?: unknown } | undefined;
  const reason = incomplete?.reason;
  if (
    typeof reason === "string" &&
    (INCOMPLETE_REASONS as readonly string[]).includes(reason)
  ) {
    return { reason: reason as IncompleteReason };
  }
  return null;
}

const INCOMPLETE_LABELS: Record<IncompleteReason, string> = {
  length: "Response hit the Max Tokens limit",
  cancelled: "Response stopped",
  interrupted: "Response interrupted",
};

/** The user-facing explanation of why a turn stopped. */
export function incompleteLabel(reason: IncompleteReason): string {
  return INCOMPLETE_LABELS[reason];
}

/**
 * Drop text the continuation repeated from the end of the partial. Local models
 * continue token-exactly, but a provider that ignores assistant prefill can restate
 * the last few words. Only a suffix the continuation opens with is removed.
 */
export function stripContinuationOverlap(
  partial: string,
  continuation: string,
): string {
  if (partial.length === 0 || continuation.length === 0) {
    return continuation;
  }
  const limit = Math.min(partial.length, continuation.length, MAX_OVERLAP);
  for (let size = limit; size >= MIN_OVERLAP; size -= 1) {
    if (continuation.startsWith(partial.slice(partial.length - size))) {
      return continuation.slice(size);
    }
  }
  return continuation;
}

/**
 * True when the "continuation" is really a fresh answer. Judged on the partial's
 * opening, which a genuine continuation never reproduces; appending a restart would
 * read as a stutter, so the caller keeps it alone.
 */
export function isRestart(partial: string, continuation: string): boolean {
  const head = partial.trimStart().slice(0, RESTART_PROBE);
  if (head.length < RESTART_PROBE) {
    // Too short to tell a restart from a coincidence.
    return false;
  }
  return continuation.trimStart().startsWith(head);
}

/**
 * Merge a partial answer with its continuation.
 *
 * `streaming` skips the restart check, which needs more text than early chunks carry;
 * it runs once at the end.
 */
export function joinContinuation(
  partial: string,
  continuation: string,
  { streaming = false }: { streaming?: boolean } = {},
): string {
  if (!partial) {
    return continuation;
  }
  if (!streaming && isRestart(partial, continuation)) {
    return continuation;
  }
  return `${partial}${stripContinuationOverlap(partial, continuation)}`;
}

/**
 * Whether a finished MLX turn used its whole budget, i.e. stopped for length.
 *
 * MLX alone needs the inference: it reports finish_reason "stop" at the cap. Everyone
 * else reports "length" themselves, transformers included (from the token count
 * `generate` returned, which distinguishes an exhausted budget from a stop token that
 * happened to land at the cap), so inferring for them would relabel a finished answer.
 */
export function budgetImpliesTruncation({
  isMlx,
  maxTokens,
  completionTokens,
}: {
  isMlx: boolean;
  maxTokens: number | undefined;
  completionTokens: number | undefined;
}): boolean {
  if (!isMlx) {
    return false;
  }
  return (
    typeof maxTokens === "number" &&
    typeof completionTokens === "number" &&
    completionTokens >= maxTokens
  );
}

/**
 * Whether an assistant turn can be resumed at all.
 *
 * A turn that called a tool cannot: the continuation runs as a sibling, so the call and
 * its result are absent from the outbound history and the model would be asked to carry
 * on from an answer whose evidence is missing. Matches the backend guard.
 */
export function isContinuableContent(
  content: readonly unknown[] | undefined,
): boolean {
  if (!content) {
    return false;
  }
  let hasText = false;
  for (const part of content) {
    const type = (part as { type?: string })?.type;
    if (type === "text") {
      hasText = hasText || ((part as { text?: string }).text ?? "").length > 0;
      continue;
    }
    // Reasoning and citations are never replayed, so they neither block nor enable.
    if (type === "reasoning" || type === "source") {
      continue;
    }
    return false;
  }
  return hasText;
}

/**
 * The newest Gemini text-part thoughtSignature on an assistant turn. The streaming
 * adapter pins it onto the final text part; the continuation carries it so the resumed
 * turn is replayed signed rather than as bare text.
 */
export function readTextThoughtSignature(
  content: readonly unknown[] | undefined,
): string | undefined {
  if (!content) {
    return undefined;
  }
  for (let i = content.length - 1; i >= 0; i -= 1) {
    const part = content[i] as
      | { type?: string; _google_thought_signature?: unknown }
      | undefined;
    if (part?.type !== "text") {
      continue;
    }
    const signature = part._google_thought_signature;
    if (typeof signature === "string" && signature) {
      return signature;
    }
  }
  return undefined;
}

/**
 * Providers that reject a trailing assistant turn.
 *
 * Anthropic removed assistant prefill in Claude 4.6 and never allowed it with extended
 * thinking. Gemini requires a multiturn request to end in a user turn or a function
 * response. Mistral takes a prefill only on an assistant message carrying `prefix: true`,
 * which neither the outbound message type nor the backend schema has. All get the partial
 * plus an instruction turn instead, keeping the last message a user turn.
 */
const PREFILL_REJECTING_PROVIDERS = new Set(["anthropic", "gemini", "mistral"]);

export function rejectsAssistantPrefill(
  providerType: string | undefined,
): boolean {
  return providerType != null && PREFILL_REJECTING_PROVIDERS.has(providerType);
}

/**
 * Self-hosted servers the backend sends the continuation flags to.
 *
 * Mirrors `_CONTINUATION_FLAG_PROVIDERS` in `core/inference/external_provider.py`: vLLM
 * and llama-server document `continue_final_message` + `add_generation_prompt`, so they
 * resume at the exact token boundary and need no overlap repair. Every other connection
 * may ignore the trailing assistant turn and restart.
 */
const EXACT_RESUME_PROVIDERS = new Set(["vllm", "llama_cpp"]);

export function resumesExactly(providerType: string | undefined): boolean {
  return providerType != null && EXACT_RESUME_PROVIDERS.has(providerType);
}

/**
 * Whether the run this thread would start can resume a partial at all.
 *
 * These three answer from scratch before the continuation request is read: audio input
 * re-listens, an audio-output model regenerates the clip, and armed Deep Research
 * replaces the turn with a report. Continue would restart, so it is hidden.
 */
export function modeAllowsContinuation({
  fromAudioInput,
  audioOutputModel,
  deepResearchArmed,
}: {
  fromAudioInput: boolean;
  audioOutputModel: boolean;
  deepResearchArmed: boolean;
}): boolean {
  return !(fromAudioInput || audioOutputModel || deepResearchArmed);
}

/** Asks for a continuation when the partial cannot be sent as a prefill. */
export const CONTINUE_INSTRUCTION =
  "Continue your previous response from exactly where it stopped. " +
  "Do not repeat any text you already wrote and do not restate the answer.";

/** The `runConfig.custom` key carrying a continuation request to the chat adapter. */
export const CONTINUATION_RUN_CONFIG_KEY = "unslothContinuation";

export type ContinuationRequest = {
  /** The partial answer to resume, exactly as it was rendered. */
  partial: string;
  /**
   * Gemini text-part thoughtSignature from the turn being resumed. The sibling run drops
   * the original assistant message, so replaying it here keeps the history signed.
   */
  thoughtSignature?: string;
};

/** Read a continuation request out of a run's `runConfig`, if it is one. */
export function readContinuationRequest(
  runConfig: unknown,
): ContinuationRequest | null {
  const custom = (runConfig as { custom?: Record<string, unknown> } | undefined)
    ?.custom;
  const request = custom?.[CONTINUATION_RUN_CONFIG_KEY] as
    | { partial?: unknown; thoughtSignature?: unknown }
    | undefined;
  const partial = request?.partial;
  if (typeof partial === "string" && partial.length > 0) {
    const signature = request?.thoughtSignature;
    return typeof signature === "string" && signature
      ? { partial, thoughtSignature: signature }
      : { partial };
  }
  return null;
}
