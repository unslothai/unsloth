// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether Max Tokens, not the context window, stopped generation: both yield `finish_reason:
 * "length"`. With Max Tokens on "Max" the backend sends the full context length, so a cap equal
 * to it reads as unset. `promptTokens` is the server's count; absent, the cap alone decides.
 */
export function maxTokensIsTheLimit({
  cap,
  contextLength,
  promptTokens,
}: {
  cap: number | null;
  contextLength: number | null;
  promptTokens: number | null;
}): boolean {
  const window = contextLength ?? Number.POSITIVE_INFINITY;
  if (cap === null || cap >= window) {
    return false;
  }
  if (promptTokens === null) {
    return true;
  }
  // Strictly below: at equality the context wall binds in the same token.
  return promptTokens + cap < window;
}

/** `context_length` is local; `context_window` is external; `unknown` lacks evidence. */
export type LengthStopCause =
  | "max_tokens"
  | "context_length"
  | "context_window"
  | "unknown";

export function lengthStopCause({
  cap,
  contextLength,
  promptTokens,
  completionTokens,
}: {
  cap: number | null;
  contextLength: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
}): LengthStopCause {
  // Only pass counts accepted by windowEvidenceCount for an unknown window.
  if (contextLength === null && cap !== null) {
    if (completionTokens === null) {
      return "unknown";
    }
    return completionTokens < cap ? "context_window" : "max_tokens";
  }
  return maxTokensIsTheLimit({ cap, contextLength, promptTokens })
    ? "max_tokens"
    : "context_length";
}

/** Save external window exhaustion separately to prevent automatic continuation. */
export function lengthIncompleteReason(
  cause: LengthStopCause,
): "context_window" | "length" {
  return cause === "context_window" ? "context_window" : "length";
}

/** llama.cpp counts include reasoning. Ignore usage counts: providers may exclude reasoning
 *  or enforce a lower output cap, so those counts cannot establish context exhaustion. */
export function windowEvidenceCount(
  timings: { predicted_n?: unknown } | null | undefined,
): number | null {
  const count = timings?.predicted_n;
  return typeof count === "number" && Number.isFinite(count) ? count : null;
}
