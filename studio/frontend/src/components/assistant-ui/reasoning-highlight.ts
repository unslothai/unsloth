// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HighlightResult } from "@streamdown/code";

export type ReasoningHighlightRequest = {
  client: number;
  revision: number;
  source: string | { from: number; text: string };
  language: string | null;
  lines: number[];
};

/** A client sends its complete source once, then only newly appended bytes. */
export function reasoningHighlightSource(
  previous: string,
  source: ReasoningHighlightRequest["source"],
): string {
  return typeof source === "string"
    ? source
    : previous.slice(0, source.from) + source.text;
}
export type ReasoningHighlightReply = {
  client: number;
  revision: number;
  lines: { line: number; tokens: HighlightResult["tokens"][number] }[];
};

/** Transfer only requested lines; the worker keeps the complete grammar context. */
export function reasoningHighlightReply(
  request: ReasoningHighlightRequest,
  result: HighlightResult | null,
): ReasoningHighlightReply {
  return {
    client: request.client,
    revision: request.revision,
    lines: result
      ? request.lines.flatMap((line) => {
          const tokens = result.tokens[line];
          return tokens ? [{ line, tokens }] : [];
        })
      : [],
  };
}
