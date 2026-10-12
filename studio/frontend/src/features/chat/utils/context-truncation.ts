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

/** Whether the fit removed turns from the prompt that was actually sent. Not `fits`: a fit under
 *  the physical window that misses the reply reserve still sends the shortened prompt with
 *  `fits: false`, and those turns are just as gone. Paths returning the ORIGINAL messages
 *  report `dropped_messages` as zero. */
export function promptWasShortened(
  truncation: ContextTruncation | undefined,
): truncation is ContextTruncation {
  return (truncation?.dropped_messages ?? 0) > 0;
}

export function compactionBoundary(
  truncation: ContextTruncation | undefined,
): number {
  if (!promptWasShortened(truncation)) return 0;
  // boundary_messages is where the boundary sits in the saved transcript, and every fit that
  // evicts records one. So the fallback is only for turns saved before it existed, hence gated
  // on `fits`: elsewhere dropped_messages is a per-refit total, not a position, and reading it
  // as one sets a high-water mark `showsNotice` never sees exceeded again.
  return (
    truncation.boundary_messages ??
    (truncation.fits ? (truncation.dropped_messages ?? 0) : 0)
  );
}

/** A saved-transcript boundary can stay put when a tool loop compacts its own messages. */
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

/** One message as it was sent, and the transcript row it came from (none for a synthetic one). */
export type SentMessageRow = { readonly role: string; readonly rowId?: string };

/**
 * The transcript row the fit's boundary sits above, or null when it cannot be placed exactly.
 *
 * `sent` is the request's non-system messages in order. The backend counts on its own copy,
 * which the route may have coalesced or pruned, so the position is trusted only when both
 * sides agree on how many messages precede the newest user turn. A boundary inside a row (a
 * tool exchange evicted from the middle of one reply) moves down to the next row, since the
 * divider can only sit between rows.
 */
export function compactionBoundaryRowId(
  truncation: ContextTruncation,
  sent: readonly SentMessageRow[],
): string | null {
  const position = truncation.boundary_position;
  const preceding = truncation.boundary_branch_messages;
  if (position === undefined) return null;
  let cutoff = Math.max(0, sent.length - 1);
  for (let index = sent.length - 1; index >= 0; index -= 1) {
    if (sent[index]?.role === "user") {
      cutoff = index;
      break;
    }
  }
  if (
    cutoff !== preceding ||
    !Number.isSafeInteger(position) ||
    position < 1 ||
    position > cutoff
  ) {
    return null;
  }
  const above = sent[position - 1]?.rowId;
  for (let index = position; index <= cutoff; index += 1) {
    const rowId = sent[index]?.rowId;
    if (rowId === undefined) return null;
    if (rowId !== above) return rowId;
  }
  return null;
}

function nonNegativeInt(value: number | undefined): number {
  // A propagated NaN would print "NaN tokens on its own" at the user.
  return Number.isFinite(value) ? Math.max(0, Math.trunc(value as number)) : 0;
}

export function latestTurnOwnTokens(
  truncation: ContextTruncation | null | undefined,
): number {
  // `latest_turn_tokens` prices a whole rendered PROMPT: template wrapper plus, on a tool-enabled
  // request, the entire tool catalogue. That sits inside `irreducible_tokens` too and does not
  // cancel (the built-in catalogue alone is over a thousand tokens), so a 6-token "hi" was
  // reported as thousands. `shared_prompt_tokens` is that floor, measured on an empty prompt.
  const latest = nonNegativeInt(truncation?.latest_turn_tokens);
  // Never the whole turn: reporting it as zero tokens is a worse lie than the old one.
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
  // `latest_turn_exact: false` means nothing could price the turn, so the number is the message's
  // JSON at four characters a token while every other number here is a tokenizer count of a
  // rendered prompt: 16,400 characters of newlines estimate 8,207 against 557 rendered. A turn
  // the template renders as nothing is NOT this case. Absent means a server predating the flag.
  if (!(truncation.latest_turn_exact ?? true)) return false;
  // Measured WITHOUT the shared floor, so a turn only over budget once a tool catalogue stands
  // beside it is not blamed. An older server sends no floor, which reads as zero.
  return latestTurnOwnTokens(truncation) > budget;
}

export function historyCannotHelp(
  truncation: ContextTruncation | null | undefined,
): boolean {
  if (!truncation) return false;
  // `irreducible_tokens` prices what survives dropping every evictable group: template wrapper,
  // tool catalogue, system turns and the newest turn. At or over the WINDOW, llama-server
  // refuses on size alone however short the conversation gets, so "start a new chat" opens a
  // chat that fails identically. Below it, shortening really can work.
  // The fit refuses at `prompt_target` but passes the untrimmed messages on.
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
    // A turn can compact more than once (the tool loop refits per iteration), so these accumulate
    // rather than taking the last chunk's value. Spread conditionally so a plain rolling-window
    // response keeps its old shape.
    ...spreadSum("archived_messages", current.archived_messages, incoming.archived_messages),
    ...spreadSum("recalled_chunks", current.recalled_chunks, incoming.recalled_chunks),
  };

  // boundary_messages needs no rule: it is absolute, so the spread above already keeps the latest
  // fit's value. Summing it is the bug it exists to fix. boundary_anchor rides along for the
  // same reason, and the two must come from the SAME fit.

  // The irreducible diagnosis describes ONE fit that gave up, so an earlier failure followed by a
  // later success would otherwise leave those numbers on a result that fit. Deleted rather than
  // spread as undefined, which would put both keys on every ordinary response.
  if (incoming.fits) {
    delete merged.irreducible_tokens;
    delete merged.latest_turn_tokens;
    // Rides with the count it describes: alone it says nothing, and left behind it would describe a
    // number that is no longer there.
    delete merged.latest_turn_exact;
    // Likewise the floor: a stale one subtracted from a later fit's count moves the blame rather than removing it.
    delete merged.shared_prompt_tokens;
  }
  return merged;
}
