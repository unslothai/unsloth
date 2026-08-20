// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

/** Restore assistant-ui's status without losing the product-specific stop reason. */
export function restoredAssistantStatus(
  metadata: unknown,
): import("@assistant-ui/react").MessageStatus {
  const incomplete = readIncompleteInfo(metadata);
  if (!incomplete) {
    return { type: "complete", reason: "unknown" };
  }
  return {
    type: "incomplete",
    reason: incomplete.reason === "cancelled" ? "cancelled" : "error",
  };
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

/**
 * Resuming a Max Tokens cut WITHOUT asking.
 *
 * Hitting the cap is not a decision the user made; it is the reply running out of room
 * mid-sentence, and the only sensible answer to "the answer is not finished" is to finish
 * it. Every other reason is left alone: `cancelled` is the user pressing Stop, so
 * resuming would restart the thing they just stopped, and `interrupted` means the
 * connection dropped, where a silent retry can hide a broken link.
 *
 * Bounded, because a model that will not stop would otherwise loop forever, and each
 * round grows the transcript and drives compaction harder. After the budget is spent the
 * bar comes back and the user decides.
 */
export const AUTO_CONTINUE_LIMIT = 3;

/**
 * Rounds already spent per logical turn, keyed by the parent the continuation hangs off.
 *
 * Keyed on the PARENT, not the message: a continuation runs as a sibling of the turn it
 * resumes, so each round produces a new message id and a per-message counter would reset
 * every time and never reach its limit. The parent is the one id every round of the same
 * turn shares. In memory only; a reload is a fresh decision by a present user.
 */
const spent = new Map<string, number>();

/**
 * Whether this cut should resume on its own.
 *
 * `fits` and `promptTarget` come from the resumed turn's own truncation metadata, and
 * both exist because resuming FIGHTS the context window. A continuation replays the
 * partial as the final assistant turn, and the fit protects the final group, so the
 * partial is the one thing compaction may not evict. Once it is large enough that the
 * system turn plus the carried-forward block will not fit beside it, the request is
 * irreducible and every further round produces the same refusal.
 *
 * Measured at a 4,864-token context: a 3,217-token partial plus system and X came to
 * 4,218 against a 3,648-token target, so the answer could not be resumed at all -- and
 * three automatic rounds each asked anyway and each failed identically.
 */
export function shouldAutoContinue(
  reason: IncompleteReason | null | undefined,
  key: string | null | undefined,
  {
    limit = AUTO_CONTINUE_LIMIT,
    fits,
    partialTokens,
    promptTarget,
  }: {
    limit?: number;
    /** `contextTruncation.fits` of the turn being resumed. */
    fits?: boolean;
    /** Estimated size of the partial that would be replayed. */
    partialTokens?: number;
    /** `contextTruncation.prompt_target`: what the window leaves for the prompt. */
    promptTarget?: number;
  } = {},
): boolean {
  if (reason !== "length" || !key) {
    return false;
  }
  if (fits === false) {
    // This turn's own fit was already refused. Resuming re-sends a partial that is only
    // ever longer, so the next round cannot fit either.
    return false;
  }
  if (
    typeof partialTokens === "number" &&
    typeof promptTarget === "number" &&
    promptTarget > 0 &&
    partialTokens >= promptTarget
  ) {
    // The partial alone meets the budget, so nothing can be sent beside it -- not the
    // system prompt, not the user's own question. Raising Context Length is the only
    // remedy and the bar says so.
    return false;
  }
  return (spent.get(key) ?? 0) < limit;
}

/** Record a round against `key`. Called before the run starts, so a run that fails to
 * produce anything still consumes its budget rather than retrying forever. */
export function recordAutoContinue(key: string): void {
  spent.set(key, (spent.get(key) ?? 0) + 1);
}

/** Rounds spent so far, for the indicator and for tests. */
export function autoContinueCount(key: string | null | undefined): number {
  return key ? (spent.get(key) ?? 0) : 0;
}

/**
 * Messages this session has already continued automatically, by message id.
 *
 * Separate from `spent`, which counts rounds per logical turn and is what bounds a
 * runaway loop. This answers a different question: has THIS message already been
 * resumed once. A component-local ref cannot, because it dies with the component.
 * Leave the chat with a truncated branch selected and come back, and the ref is fresh
 * while the parent still has budget, so the effect fires again and creates another
 * sibling and another paid request. Module scope outlives the remount; the ids are
 * short and bounded by how many replies a session truncates.
 */
const continued = new Set<string>();

/** Whether `messageId` still needs continuing. False once it has been claimed. */
export function claimAutoContinue(messageId: string | null | undefined): boolean {
  if (!messageId || continued.has(messageId)) {
    return false;
  }
  continued.add(messageId);
  return true;
}

/** Whether `messageId` was already continued automatically this session. */
export function wasAutoContinued(messageId: string | null | undefined): boolean {
  return Boolean(messageId && continued.has(messageId));
}

/**
 * Whether THIS message is the one to continue automatically.
 *
 * `shouldAutoContinue` answers about the turn: is the reason right, does the partial
 * still fit, is the round budget unspent. It keeps saying yes after a message has been
 * claimed, because the budget is per turn and one spent round out of three leaves the
 * turn resumable, while the claim is per message and is what actually decides whether a
 * run starts. Anything rendering off the turn's answer alone therefore reports an
 * already-claimed message as continuing while `claimAutoContinue` refuses to start
 * anything: a spinner that never resolves, over the manual Continue button it replaces.
 *
 * Reachable without a reload. The continuation runs as a sibling and the branch picker
 * leads straight back to the truncated partial, and leaving the chat and returning lands
 * on it too whenever it is still the selected branch.
 */
export function shouldAutoContinueMessage(
  messageId: string | null | undefined,
  reason: IncompleteReason | null | undefined,
  key: string | null | undefined,
  options: Parameters<typeof shouldAutoContinue>[2] = {},
): boolean {
  if (wasAutoContinued(messageId)) {
    return false;
  }
  return shouldAutoContinue(reason, key, options);
}

/** Test seam; also lets a new thread start from zero. */
export function resetAutoContinue(key?: string): void {
  if (key === undefined) {
    spent.clear();
    continued.clear();
  } else {
    spent.delete(key);
  }
}
