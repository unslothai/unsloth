// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Deliberately not imported from features/chat: the pill bundle must stay
// tiny and chat-api drags the full chat page graph with it.

// The auth barrel drags the login/change-password pages (~156 kB) into the
// pill window's load graph, so import the leaf module directly.
// eslint-disable-next-line no-restricted-imports
import { authFetch } from "@/features/auth/api";

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
};

export async function* streamCompletion(
  payload: StreamRequest,
  signal: AbortSignal,
): AsyncGenerator<string> {
  const response = await authFetch("/v1/chat/completions", {
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
  // A dropped connection ends the body exactly like a finished stream, so
  // without this an answer cut mid-sentence is rendered as a complete one.
  let terminated = false;

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
          if (data === "[DONE]") return;
          let chunk: ChatChunk;
          try {
            chunk = JSON.parse(data) as ChatChunk;
          } catch {
            continue;
          }
          if (chunk.choices?.[0]?.finish_reason) terminated = true;
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) yield delta;
        }
        separatorIndex = buffer.search(/\r?\n\r?\n/);
      }
    }
    if (!terminated) {
      throw new Error("Stream ended before the answer was complete");
    }
  } finally {
    reader.releaseLock();
  }
}
