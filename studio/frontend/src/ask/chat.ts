// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Leaf imports: the feature barrels pull the full app into this small window.
// eslint-disable-next-line no-restricted-imports
import { authFetch } from "@/features/auth/api";
// eslint-disable-next-line no-restricted-imports
import { assertCompletedPaddedBody } from "@/features/chat/api/padded-response";
import { BACKEND_PORT_STORAGE_KEY, setApiBase } from "@/lib/api-base";

export type ChatMessage = { role: "user" | "assistant"; content: string };

export class AskError extends Error {
  readonly kind: "noModel" | "failed";

  constructor(kind: "noModel" | "failed") {
    super(kind);
    this.kind = kind;
  }
}

/** Points this window at the backend the main window found. False when none is known yet. */
export function adoptBackendPort(): boolean {
  let port = 0;
  try {
    port = Number(localStorage.getItem(BACKEND_PORT_STORAGE_KEY));
  } catch {
    return false;
  }
  if (!Number.isInteger(port) || port <= 0 || port > 65535) return false;
  setApiBase(port);
  return true;
}

async function getJson<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await authFetch(path, { signal });
  if (!response.ok) throw new Error(`${path} failed (${response.status})`);
  return (await response.json()) as T;
}

/** The loaded model, else the last one loaded in Chat, loading it first. */
export async function resolveModel(
  signal: AbortSignal,
  onLoading: (model: string) => void,
): Promise<string> {
  const status = await getJson<{ active_model: string | null }>(
    "/api/inference/status",
    signal,
  );
  if (status.active_model) return status.active_model;
  const last = await getJson<{ id?: string | null; gguf_variant?: string | null }>(
    "/api/settings/last-local-model",
    signal,
  );
  if (!last.id) throw new AskError("noModel");
  onLoading(last.id);
  const response = await authFetch("/api/inference/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_path: last.id, gguf_variant: last.gguf_variant ?? undefined }),
    signal,
  });
  // The route pads its body to outlive proxies, so only the finished body says the load succeeded.
  const body = (await response.json().catch(() => null)) as {
    _deferred_error?: unknown;
  } | null;
  if (!response.ok || body?._deferred_error) throw new AskError("failed");
  assertCompletedPaddedBody(body, "Model load");
  return last.id;
}

/** Streams the answer's text; throws unless the stream reaches a clean stop. */
export async function* streamAnswer(
  model: string,
  messages: ChatMessage[],
  signal: AbortSignal,
): AsyncGenerator<string> {
  const response = await authFetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, stream: true }),
    signal,
  });
  if (!response.ok || !response.body) throw new AskError("failed");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;
  let sawContent = false;
  // Reasoning-only replies (thinking models with no visible answer) are shown instead of nothing.
  let reasoning = "";
  try {
    while (!finished) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") {
          finished = true;
          break;
        }
        let chunk: {
          error?: unknown;
          choices?: Array<{
            delta?: { content?: string | null; reasoning_content?: string | null };
            finish_reason?: string | null;
          }>;
        };
        try {
          chunk = JSON.parse(data);
        } catch {
          continue;
        }
        if (chunk.error) throw new AskError("failed");
        const choice = chunk.choices?.[0];
        // "length" and "content_filter" are clipped answers, not whole ones.
        if (choice?.finish_reason && choice.finish_reason !== "stop") {
          throw new AskError("failed");
        }
        if (choice?.delta?.content) {
          sawContent = true;
          yield choice.delta.content;
        } else if (choice?.delta?.reasoning_content) {
          reasoning += choice.delta.reasoning_content;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  // A dropped connection looks like the end of the stream: require [DONE].
  if (!finished) throw new AskError("failed");
  if (!sawContent && reasoning) yield reasoning;
}
