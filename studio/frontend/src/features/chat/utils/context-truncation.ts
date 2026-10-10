// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { OpenAIChatChunk } from "../types/api";

export type ContextTruncation = NonNullable<
  OpenAIChatChunk["context_truncated"]
>;

function spreadSum(
  key: "archived_messages" | "recalled_chunks",
  a: number | undefined,
  b: number | undefined,
): Record<string, number> {
  if (a === undefined && b === undefined) return {};
  return { [key]: (a ?? 0) + (b ?? 0) };
}

/** Not `fits`: a fit missing the reply reserve still sends the shortened prompt. */
export function promptWasShortened(
  truncation: ContextTruncation | undefined,
): truncation is ContextTruncation {
  return (truncation?.dropped_messages ?? 0) > 0;
}

export function compactionBoundary(
  truncation: ContextTruncation | undefined,
): number {
  if (!promptWasShortened(truncation)) return 0;
  // Fallback only for turns saved before boundary_messages existed; elsewhere it is a per-refit total.
  return (
    truncation.boundary_messages ??
    (truncation.fits ? (truncation.dropped_messages ?? 0) : 0)
  );
}

export function shouldShowCompactionNotice(
  truncation: ContextTruncation | undefined,
  previousBoundary: number,
): boolean {
  return (
    promptWasShortened(truncation) &&
    (truncation.checkpoint_started === true ||
      compactionBoundary(truncation) > previousBoundary)
  );
}

function nonNegativeInt(value: number | undefined): number {
  return Number.isFinite(value) ? Math.max(0, Math.trunc(value as number)) : 0;
}

export function latestTurnOwnTokens(
  truncation: ContextTruncation | null | undefined,
): number {
  // latest_turn_tokens includes the template and tool catalogue; subtract the shared_prompt_tokens floor.
  const latest = nonNegativeInt(truncation?.latest_turn_tokens);
  const shared = Math.min(
    nonNegativeInt(truncation?.shared_prompt_tokens),
    Math.max(0, latest - 1),
  );
  return latest - shared;
}

export function latestTurnIsTheProblem(
  truncation: ContextTruncation | null | undefined,
  budget: number,
): boolean {
  if (!truncation) return false;
  // latest_turn_exact false means a chars/4 estimate, not comparable to tokenizer counts.
  if (!(truncation.latest_turn_exact ?? true)) return false;
  return latestTurnOwnTokens(truncation) > budget;
}

export function historyCannotHelp(
  truncation: ContextTruncation | null | undefined,
): boolean {
  if (!truncation) return false;
  // At or over the window, llama-server refuses whatever is dropped, so a new chat would fail too.
  const irreducible = nonNegativeInt(truncation.irreducible_tokens);
  const window = nonNegativeInt(truncation.context_length);
  return irreducible > 0 && window > 0 && irreducible >= window;
}

export function mergeContextTruncation(
  current: ContextTruncation | undefined,
  incoming: ContextTruncation,
): ContextTruncation {
  if (!current) return incoming;

  const merged = {
    ...current,
    ...incoming,
    dropped_messages: current.dropped_messages + incoming.dropped_messages,
    // Sticky: a later replay fit in the same tool loop must not clear it.
    ...(current.checkpoint_started !== undefined ||
    incoming.checkpoint_started !== undefined
      ? {
          checkpoint_started:
            current.checkpoint_started === true ||
            incoming.checkpoint_started === true,
        }
      : {}),
    prompt_tokens_before:
      current.prompt_tokens_before ?? incoming.prompt_tokens_before,
    prompt_tokens_after:
      incoming.prompt_tokens_after ?? current.prompt_tokens_after,
    // A turn can compact more than once per tool loop, so these accumulate.
    ...spreadSum("archived_messages", current.archived_messages, incoming.archived_messages),
    ...spreadSum("recalled_chunks", current.recalled_chunks, incoming.recalled_chunks),
  };

  // boundary_messages is absolute: keep the latest fit's value, with boundary_anchor from the same fit.

  // The irreducible diagnosis describes one failed fit, so drop it once a later fit succeeds.
  if (incoming.fits) {
    delete merged.irreducible_tokens;
    delete merged.latest_turn_tokens;
    delete merged.latest_turn_exact;
    delete merged.shared_prompt_tokens;
  }
  return merged;
}
