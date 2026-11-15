// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const MIN_RUN = 120;

function isBase64Char(c: number): boolean {
  return (
    (c >= 97 && c <= 122) ||
    (c >= 65 && c <= 90) ||
    (c >= 48 && c <= 57) ||
    c === 43 ||
    c === 47
  );
}

/** text.replace(/[A-Za-z0-9+/]{120,}={0,2}/g, " ") in one pass; the regex retries inside every word. */
export function stripBase64Runs(text: string): string {
  let pieces: string[] | null = null;
  let last = 0;
  let i = 0;
  const n = text.length;
  while (i < n) {
    if (!isBase64Char(text.charCodeAt(i))) {
      i++;
      continue;
    }
    const start = i;
    while (i < n && isBase64Char(text.charCodeAt(i))) i++;
    if (i - start < MIN_RUN) continue;
    for (let pad = 0; pad < 2 && i < n && text.charCodeAt(i) === 61; pad++) i++;
    pieces ??= [];
    pieces.push(text.slice(last, start), " ");
    last = i;
  }
  return pieces === null ? text : pieces.join("") + text.slice(last);
}

// A lone space needs no rewrite, so only runs that change match.
const WHITESPACE_TO_COLLAPSE = / \s+|[^\S ]\s*/g;

/** Same result as text.replace(/\s+/g, " "), without rewriting every single space. */
export function collapseWhitespace(text: string): string {
  return text.replace(WHITESPACE_TO_COLLAPSE, " ");
}
