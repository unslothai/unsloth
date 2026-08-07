// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Resuming a response that stopped early.
 *
 * A turn can end before the model was done in three ways the user can act on: Max
 * Tokens ran out (`length`), Stop was pressed (`cancelled`), or the stream was cut
 * (`interrupted`). Continuing re-sends the conversation with the partial as the final
 * assistant turn plus `continue_final_message`, so the prompt ends mid-sentence and
 * the model emits the next token. The new text is appended to the partial.
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
 * Drop text the continuation repeated from the end of the partial.
 *
 * Local models continue token-exactly, but a provider that ignores assistant prefill
 * can restate the last few words. Only a suffix the continuation opens with is removed.
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
 * True when the "continuation" is really a fresh answer.
 *
 * Judged on the partial's opening, which a genuine continuation never reproduces.
 * Appending a restart would read as a stutter, so the caller keeps it alone.
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

/** The `runConfig.custom` key carrying a continuation request to the chat adapter. */
export const CONTINUATION_RUN_CONFIG_KEY = "unslothContinuation";

export type ContinuationRequest = {
  /** The partial answer to resume, exactly as it was rendered. */
  partial: string;
};

/** Read a continuation request out of a run's `runConfig`, if it is one. */
export function readContinuationRequest(
  runConfig: unknown,
): ContinuationRequest | null {
  const custom = (runConfig as { custom?: Record<string, unknown> } | undefined)
    ?.custom;
  const request = custom?.[CONTINUATION_RUN_CONFIG_KEY] as
    | { partial?: unknown }
    | undefined;
  const partial = request?.partial;
  if (typeof partial === "string" && partial.length > 0) {
    return { partial };
  }
  return null;
}
