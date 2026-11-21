// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HighlightResult } from "@streamdown/code";

export type TokenLine = HighlightResult["tokens"][number];

const MAX_ENTRIES = 128;
const MAX_CHARACTERS = 1_000_000;
const results = new Map<string, { result: HighlightResult; chars: number }>();
let characters = 0;

const keyOf = (language: string | null, source: string): string =>
  `${language ?? ""}\u0000${source}`;

export function cachedSettledColour(
  language: string | null,
  source: string,
): HighlightResult | null {
  return results.get(keyOf(language, source))?.result ?? null;
}

export function rememberSettledColour(
  language: string | null,
  source: string,
  result: HighlightResult,
): void {
  if (source.length > MAX_CHARACTERS) return;
  const key = keyOf(language, source);
  const held = results.get(key);
  if (held) {
    characters -= held.chars;
    results.delete(key);
  }
  results.set(key, { result, chars: source.length });
  characters += source.length;
  while (results.size > MAX_ENTRIES || characters > MAX_CHARACTERS) {
    const oldest = results.entries().next();
    if (oldest.done) break;
    results.delete(oldest.value[0]);
    characters -= oldest.value[1].chars;
  }
}

/** The tokens of `line` covering `text` from `column`; null unless they spell it exactly. */
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

const spells = (line: TokenLine | undefined, text: string): boolean => {
  if (line === undefined) return false;
  let at = 0;
  for (const token of line) {
    if (!text.startsWith(token.content, at)) return false;
    at += token.content.length;
  }
  return at === text.length;
};

const sameStyle = (
  a: Record<string, string> | undefined,
  b: Record<string, string> | undefined,
): boolean => {
  if (a === b) return true;
  if (!a || !b) return false;
  const keys = Object.keys(a);
  return (
    keys.length === Object.keys(b).length && keys.every((key) => a[key] === b[key])
  );
};

const sameLine = (a: TokenLine, b: TokenLine): boolean =>
  a.length === b.length &&
  a.every(
    (token, index) =>
      token.content === b[index].content &&
      token.color === b[index].color &&
      token.bgColor === b[index].bgColor &&
      token.fontStyle === b[index].fontStyle &&
      sameStyle(token.htmlStyle, b[index].htmlStyle),
  );

export function keepUnchangedLines(
  previous: HighlightResult | null,
  next: HighlightResult,
): HighlightResult {
  if (previous === null) return next;
  return {
    ...next,
    tokens: next.tokens.map((line, index) => {
      const old = previous.tokens[index];
      return old !== undefined && sameLine(old, line) ? old : line;
    }),
  };
}

const plainLine = (content: string): TokenLine =>
  [{ content, offset: 0 }] as unknown as TokenLine;

/** Colours from an earlier, shorter source where they still spell the line; plain otherwise. */
export function spellLines(
  result: HighlightResult,
  lines: readonly string[],
): HighlightResult {
  return {
    ...result,
    tokens: lines.map((text, index) =>
      spells(result.tokens[index], text) ? result.tokens[index] : plainLine(text),
    ),
  };
}
