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

export function mergeContextTruncation(
  current: ContextTruncation | undefined,
  incoming: ContextTruncation,
): ContextTruncation {
  if (!current) return incoming;

  return {
    ...current,
    ...incoming,
    dropped_messages: current.dropped_messages + incoming.dropped_messages,
    prompt_tokens_before:
      current.prompt_tokens_before ?? incoming.prompt_tokens_before,
    prompt_tokens_after:
      incoming.prompt_tokens_after ?? current.prompt_tokens_after,
    // A turn can compact more than once (the tool loop refits per iteration), so these
    // accumulate like dropped_messages rather than taking the last chunk's value. Spread
    // conditionally so a plain rolling-window response keeps exactly the shape it had
    // before this feature existed, with no archive keys set to undefined.
    ...spreadSum("archived_messages", current.archived_messages, incoming.archived_messages),
    ...spreadSum("recalled_chunks", current.recalled_chunks, incoming.recalled_chunks),
  };
}
