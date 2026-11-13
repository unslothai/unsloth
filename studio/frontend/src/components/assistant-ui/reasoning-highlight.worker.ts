// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createCodePlugin, normalizeLanguage } from "./code-plugin";
import { unslothDarkTheme, unslothLightTheme } from "./code-themes";
import {
  orderHighlightRequests,
  reasoningHighlightFailure,
  reasoningHighlightReply,
  reasoningHighlightSource,
  type ReasoningHighlightCommand,
  type ReasoningHighlightReady,
  type ReasoningHighlightRequest,
} from "./reasoning-highlight";

const FULL_REPLY_GUARD_MS = 15_000;
const themes = [unslothLightTheme, unslothDarkTheme] as const;
const highlighter = createCodePlugin({ themes: [...themes] });
const pending = new Map<number, ReasoningHighlightRequest>();
const revisions = new Map<number, number>();
const sources = new Map<number, string>();
let scheduled = false;

self.onmessage ??= ({ data }: MessageEvent<ReasoningHighlightCommand>) => {
  if ("cancel" in data) {
    pending.delete(data.cancel);
    revisions.delete(data.cancel);
    sources.delete(data.cancel);
    return;
  }
  const source = reasoningHighlightSource(
    sources.get(data.client) ?? "",
    data.source,
  );
  sources.set(data.client, source);
  pending.set(data.client, { ...data, source });
  revisions.set(data.client, data.revision);
  if (scheduled) return;
  scheduled = true;
  // If tokenization was busy, queued appends coalesce before the next pass.
  setTimeout(() => {
    scheduled = false;
    const requests = orderHighlightRequests([...pending.values()]);
    pending.clear();
    for (const request of requests) {
      const current = () =>
        revisions.get(request.client) === request.revision;
      let guard: ReturnType<typeof setTimeout> | null = null;
      const forget = () => {
        pending.delete(request.client);
        revisions.delete(request.client);
        sources.delete(request.client);
        if (guard !== null) clearTimeout(guard);
        guard = null;
      };
      const publish = (
        result: Parameters<typeof reasoningHighlightReply>[1],
      ) => {
        if (!current()) return;
        try {
          self.postMessage(reasoningHighlightReply(request, result));
          if (request.full && result !== null) forget();
        } catch {
          self.postMessage(
            reasoningHighlightFailure(request.client, request.revision),
          );
          forget();
        }
      };
      try {
        const options: Parameters<typeof highlighter.highlightExact>[0] = {
          code: request.source as string,
          language: normalizeLanguage(request.language ?? "text"),
          themes: [...themes],
        };
        const result = request.full
          ? highlighter.highlightExact(options, publish)
          : highlighter.highlight(options, publish);
        publish(result);
        if (request.full && result === null && current()) {
          guard = setTimeout(() => {
            guard = null;
            if (!current()) return;
            self.postMessage(
              reasoningHighlightFailure(request.client, request.revision),
            );
            forget();
          }, FULL_REPLY_GUARD_MS);
        }
      } catch {
        if (current()) {
          self.postMessage(
            reasoningHighlightFailure(request.client, request.revision),
          );
        }
      }
    }
  }, 0);
};

self.postMessage({ ready: true } satisfies ReasoningHighlightReady);
