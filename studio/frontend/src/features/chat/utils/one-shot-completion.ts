// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { streamChatCompletions } from "../api/chat-api";
import { buildTitleRequest } from "./chat-title";
import { extractDeltaText } from "./parse-assistant-content";

/**
 * One system + user turn on `checkpoint` (a local model or a connected provider) with tools and
 * thinking off, outside any chat. The title request already routes both kinds and gates their
 * sampling fields, so this only lifts its token cap. Null when the connection is unavailable.
 */
export async function streamOneShotCompletion({
  checkpoint,
  systemPrompt,
  prompt,
  maxTokens,
  signal,
  onText,
}: {
  checkpoint: string;
  systemPrompt: string;
  prompt: string;
  maxTokens: number;
  signal: AbortSignal;
  /** The raw text so far, after each delta. */
  onText?: (text: string) => void;
}): Promise<{ text: string; finishReason: string | null } | null> {
  const base = await buildTitleRequest(checkpoint, prompt, systemPrompt);
  if (!base) return null;
  // A provider's own minimum can sit above the caller's cap.
  const request = { ...base, max_tokens: Math.max(base.max_tokens, maxTokens) };
  let text = "";
  let finishReason: string | null = null;
  for await (const chunk of streamChatCompletions(request, signal)) {
    const choice = chunk.choices?.[0];
    const delta = extractDeltaText(choice?.delta?.content).text;
    // A later usage chunk has no finish reason; it must not erase this one.
    finishReason = choice?.finish_reason ?? finishReason;
    if (!delta) continue;
    text += delta;
    onText?.(text);
  }
  return { text, finishReason };
}
