// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Not from features/chat: it drags the full chat page into the pill bundle.

import { pillFetch } from "./api";

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type StreamRequest = {
  model: string;
  messages: ChatMessage[];
  stream: true;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
};

type ChatChunk = {
  choices?: Array<{
    delta?: { content?: string | null; reasoning_content?: string | null };
    finish_reason?: string | null;
  }>;
  error?: string | { message?: string };
};

export async function* streamCompletion(
  payload: StreamRequest,
  signal: AbortSignal,
): AsyncGenerator<string> {
  const response = await pillFetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail =
      (body as { detail?: unknown } | null)?.detail ??
      (body as { error?: { message?: string } } | null)?.error?.message;
    throw new Error(
      typeof detail === "string" ? detail : `Request failed (${response.status})`,
    );
  }
  if (!response.body) {
    throw new Error("Stream response missing body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  // A dropped connection looks like a finished stream; require a terminal frame.
  let terminated = false;
  // Use reasoning_content only if no visible content ever arrives.
  let sawContent = false;
  let reasoning = "";
  const promoteReasoning = (): string | null =>
    !sawContent && reasoning ? reasoning : null;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let separatorIndex = buffer.search(/\r?\n\r?\n/);
      while (separatorIndex >= 0) {
        const rawEvent = buffer.slice(0, separatorIndex);
        const separatorLength = buffer[separatorIndex] === "\r" ? 4 : 2;
        buffer = buffer.slice(separatorIndex + separatorLength);

        for (const line of rawEvent.split(/\r?\n/)) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trimStart();
          if (data === "[DONE]") {
            const only = promoteReasoning();
            if (only) yield only;
            return;
          }
          let chunk: ChatChunk;
          try {
            chunk = JSON.parse(data) as ChatChunk;
          } catch {
            continue;
          }
          // Mid-stream failure arrives as an error frame then [DONE].
          if (chunk.error) {
            const message =
              typeof chunk.error === "string"
                ? chunk.error
                : chunk.error.message;
            throw new Error(message || "The model reported an error");
          }
          const choice = chunk.choices?.[0];
          const reason = choice?.finish_reason;
          if (reason) {
            // Only "stop" (or a tool hand-off) is a whole answer; "length"/"content_filter" are clipped.
            terminated = true;
            if (reason !== "stop" && reason !== "tool_calls") {
              throw new Error(`The answer stopped early (${reason})`);
            }
          }
          const delta = choice?.delta?.content;
          if (delta) {
            sawContent = true;
            yield delta;
          } else if (choice?.delta?.reasoning_content) {
            reasoning += choice.delta.reasoning_content;
          }
        }
        separatorIndex = buffer.search(/\r?\n\r?\n/);
      }
    }
    if (!terminated) {
      throw new Error("Stream ended before the answer was complete");
    }
    const only = promoteReasoning();
    if (only) yield only;
  } finally {
    reader.releaseLock();
  }
}
