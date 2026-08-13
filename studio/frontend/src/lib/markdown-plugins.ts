// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which Streamdown plugins a document actually needs.
 *
 * Wiring math and mermaid into a document that contains neither still costs a KaTeX pass and a
 * mermaid pass over every node, on the main thread, in one commit. A deep research report is
 * ordinary prose with citations, so both are usually pure waste - and the report lands exactly
 * when the run leaves "Writing the report", which is where issue #8483 froze.
 *
 * The detection lives here rather than in a component so the model card renderer and the report
 * renderer share one rule, and so it is testable without a DOM.
 */

/** `$$…$$`, `\(…\)` and `\[…\]`; single `$` is too common in prose to treat as math. */
const NEEDS_MATH = /\$\$|\\\(|\\\[/;
// Either fence, three or more of either character: CommonMark opens a fenced code block with
// backticks or tildes, and the parser hands both to the renderer as lang "mermaid". Matching
// only backticks would render a tilde-fenced diagram as a plain code block, which is a
// regression against installing the plugin unconditionally. Deliberately not anchored to the
// line start, so a fence nested in a list item still counts: over-matching costs one unused
// plugin, under-matching costs a diagram.
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
