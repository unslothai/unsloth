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

/**
 * Whether the fit removed turns from the prompt that was actually sent.
 *
 * Not `fits`, which answers a different question: a fit that lands under the physical
 * window but misses the reply reserve sends the shortened prompt with `fits: false`, and
 * those turns are just as gone from the model's view. `dropped_messages` is the signal,
 * and every path that returned the ORIGINAL messages reports it as zero.
 */
export function promptWasShortened(
  truncation: ContextTruncation | undefined,
): truncation is ContextTruncation {
  return (truncation?.dropped_messages ?? 0) > 0;
}

export function compactionBoundary(
  truncation: ContextTruncation | undefined,
): number {
  if (!promptWasShortened(truncation)) return 0;
  // boundary_messages is where the boundary sits in the saved transcript. Every fit that
  // evicts records one, rescues included, so the fallback below is only for turns saved
  // before it existed -- hence gated on `fits`, the one shape those turns have.
  //
  // dropped_messages accumulates per refit, so it is a total and not a position. Reading
  // it as one sets a high-water mark `showsNotice` never sees exceeded again, silencing
  // every later real compaction in the thread.
  return (
    truncation.boundary_messages ??
    (truncation.fits ? (truncation.dropped_messages ?? 0) : 0)
  );
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
    prompt_tokens_before:
      current.prompt_tokens_before ?? incoming.prompt_tokens_before,
    prompt_tokens_after:
      incoming.prompt_tokens_after ?? current.prompt_tokens_after,
    // A turn can compact more than once (the tool loop refits per iteration), so these
    // accumulate rather than taking the last chunk's value. Spread conditionally so a
    // plain rolling-window response keeps its old shape, with no archive keys set to
    // undefined.
    ...spreadSum("archived_messages", current.archived_messages, incoming.archived_messages),
    ...spreadSum("recalled_chunks", current.recalled_chunks, incoming.recalled_chunks),
  };

  // boundary_messages needs no rule: it is absolute, so the spread above already keeps
  // the latest fit's value. Summing it is the bug it exists to fix. boundary_anchor rides
  // along with it for the same reason, and the two must come from the SAME fit.

  // The irreducible diagnosis describes ONE fit that gave up, so an earlier failure
  // followed by a later success would otherwise leave those numbers on a result that fit.
  // Deleted rather than spread as undefined, which would put both keys on every ordinary
  // response; delete on an absent key is a no-op.
  if (incoming.fits) {
    delete merged.irreducible_tokens;
    delete merged.latest_turn_tokens;
  }
  return merged;
}
