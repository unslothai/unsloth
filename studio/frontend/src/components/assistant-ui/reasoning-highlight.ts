// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HighlightResult } from "@streamdown/code";

export type ReasoningHighlightRequest = {
  client: number;
  revision: number;
  source: string | { from: number; text: string };
  language: string | null;
  lines: number[];
  full?: boolean;
  speculative?: boolean;
};

export type ReasoningHighlightCommand =
  | ReasoningHighlightRequest
  | { cancel: number };

export type ReasoningHighlightReady = { ready: true };

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
  result?: HighlightResult | null;
  failed?: true;
};

export type ReasoningHighlightMessage =
  | ReasoningHighlightReady
  | ReasoningHighlightReply;

export const reasoningHighlightFailure = (
  client: number,
  revision: number,
): ReasoningHighlightReply => ({ client, revision, lines: [], failed: true });

export const orderHighlightRequests = <T extends { speculative?: boolean }>(
  requests: readonly T[],
): T[] => [
  ...requests.filter((request) => !request.speculative),
  ...requests.filter((request) => request.speculative),
];

/*
 * A full result as it crosses to the main thread: tokens with equal styles share one style object
 * (the clone keeps shared references, so the main thread rebuilds a handful of styles instead of
 * one per token), and no grammar state, which nothing there resumes from a worker result.
 */
export function transferableResult(result: HighlightResult): HighlightResult {
  const styles = new Map<string, Record<string, string>>();
  const shared = (style: Record<string, string>): Record<string, string> => {
    let key = "";
    for (const property in style) key += `${property}:${style[property]};`;
    const known = styles.get(key);
    if (known !== undefined) return known;
    styles.set(key, style);
    return style;
  };
  const transferred: HighlightResult = {
    ...result,
    tokens: result.tokens.map((line) =>
      line.map((token) =>
        token.htmlStyle ? { ...token, htmlStyle: shared(token.htmlStyle) } : token,
      ),
    ),
  };
  delete transferred.grammarState;
  return transferred;
}

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
    ...(request.full
      ? { result: result === null ? null : transferableResult(result) }
      : {}),
  };
}
