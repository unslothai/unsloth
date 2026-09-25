// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createCodePlugin, normalizeLanguage } from "./code-plugin";
import { unslothDarkTheme, unslothLightTheme } from "./code-themes";
import {
  reasoningHighlightReply,
  reasoningHighlightSource,
  type ReasoningHighlightRequest,
} from "./reasoning-highlight";

const themes = [unslothLightTheme, unslothDarkTheme] as const;
const highlighter = createCodePlugin({ themes: [...themes] });
const pending = new Map<number, ReasoningHighlightRequest>();
const revisions = new Map<number, number>();
const sources = new Map<number, string>();
let scheduled = false;

self.onmessage = ({
  data,
}: MessageEvent<ReasoningHighlightRequest | { cancel: number }>) => {
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
    const requests = [...pending.values()];
    pending.clear();
    for (const request of requests) {
      const publish = (
        result: Parameters<typeof reasoningHighlightReply>[1],
      ) => {
        if (revisions.get(request.client) === request.revision)
          self.postMessage(reasoningHighlightReply(request, result));
      };
      try {
        const result = highlighter.highlight(
          {
            code: request.source as string,
            language: normalizeLanguage(request.language ?? "text"),
            themes: [...themes],
          },
          publish,
        );
        publish(result);
      } catch {
        publish(null);
      }
    }
  }, 0);
};
