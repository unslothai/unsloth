// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which Streamdown plugins a document actually needs.
 *
 * Math and mermaid over a document with neither still cost a pass per node on the main thread,
 * in the one commit that lands the finished report - where issue #8483 froze. Detection lives
 * here so the model card and report renderers share one rule, testable without a DOM.
 */

/** `$$…$$`, `\(…\)` and `\[…\]`; single `$` is too common in prose to treat as math. */
const NEEDS_MATH = /\$\$|\\\(|\\\[/;
// CommonMark fences with three or more backticks or tildes, and both reach the renderer as lang
// "mermaid", so backticks alone would render a tilde-fenced diagram as a plain code block. Not
// anchored to the line start, so a fence nested in a list item still counts: over-matching costs
// one unused plugin, under-matching costs a diagram.
const NEEDS_MERMAID = /(?:`{3,}|~{3,})[ \t]*mermaid\b/;

/** Past this a document stays plain monospace: shiki is not worth the main-thread time. */
export const MAX_HIGHLIGHT_CHARS = 20_000;

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
