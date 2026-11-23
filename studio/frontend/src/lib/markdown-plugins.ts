// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Math and mermaid passes cost main-thread time even on documents without them. */

/** Single `$` is too common in prose to treat as math. */
const NEEDS_MATH = /\$\$|\\\(|\\\[/;
// Both fence styles reach the renderer as mermaid. Unanchored on purpose: over-matching is cheap.
const NEEDS_MERMAID = /(?:`{3,}|~{3,})[ \t]*mermaid\b/;

export const MAX_HIGHLIGHT_CHARS = 20_000;

export function codeFence(source: string): string {
  const longest = (source.match(/`+/g) ?? []).reduce(
    (max, run) => Math.max(max, run.length),
    0,
  );
  return "`".repeat(Math.max(3, longest + 1));
}

export interface MarkdownPluginNeeds {
  math: boolean;
  mermaid: boolean;
  code: boolean;
}

export function markdownPluginNeeds(markdown: string): MarkdownPluginNeeds {
  return {
    math: NEEDS_MATH.test(markdown),
    mermaid: NEEDS_MERMAID.test(markdown),
    code: markdown.length <= MAX_HIGHLIGHT_CHARS,
  };
}
