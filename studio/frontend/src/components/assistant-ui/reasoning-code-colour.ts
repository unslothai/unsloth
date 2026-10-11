// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HighlightResult } from "@streamdown/code";

export type TokenLine = HighlightResult["tokens"][number];

// Settled thinking fences, coloured once by the worker. Remounting a row (the trace mounts and
// unmounts chunks as the reader moves) must not ask again, so results outlive their rows.
const MAX_ENTRIES = 128;
const MAX_CHARACTERS = 1_000_000;
const results = new Map<string, HighlightResult>();
let characters = 0;

const keyOf = (language: string | null, source: string): string =>
  `${language ?? ""}\u0000${source}`;

export function cachedSettledColour(
  language: string | null,
  source: string,
): HighlightResult | null {
  // A pure read, so a render may ask; recency is kept by the write side.
  return results.get(keyOf(language, source)) ?? null;
}

export function rememberSettledColour(
  language: string | null,
  source: string,
  result: HighlightResult,
): void {
  if (source.length > MAX_CHARACTERS) return;
  const key = keyOf(language, source);
  if (!results.has(key)) characters += source.length;
  results.delete(key);
  results.set(key, result);
  while (results.size > MAX_ENTRIES || characters > MAX_CHARACTERS) {
    const oldest = results.keys().next();
    if (oldest.done) break;
    results.delete(oldest.value);
    characters -= oldest.value.length;
  }
}

/**
 * The tokens of `line` covering `text`, which starts at `column`; null when they do not spell
 * `text` exactly, so a row never shows another source's colours.
 */
export function clipLineTokens(
  line: TokenLine | undefined,
  column: number,
  text: string,
): TokenLine | null {
  if (line === undefined) return null;
  const clipped: TokenLine = [];
  let offset = 0;
  let spelled = "";
  for (const token of line) {
    const start = offset;
    offset += token.content.length;
    const content = token.content.slice(
      Math.max(0, column - start),
      Math.max(0, Math.min(token.content.length, column + text.length - start)),
    );
    if (content === "") continue;
    spelled += content;
    clipped.push(content === token.content ? token : { ...token, content });
  }
  return spelled === text ? clipped : null;
}
