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

export function compactionBoundary(
  truncation: ContextTruncation | undefined,
): number {
  if (!truncation?.fits) return 0;
  // boundary_messages is where the boundary sits in the saved transcript.
  // dropped_messages accumulates what each fit removed, so a tool-heavy turn reports far
  // more than the boundary moved and a later real advance looks like none. Fallback only,
  // for turns saved before the boundary was recorded.
  return truncation.boundary_messages ?? truncation.dropped_messages ?? 0;
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
  // the latest fit's value. Summing it is the bug it exists to fix.

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
