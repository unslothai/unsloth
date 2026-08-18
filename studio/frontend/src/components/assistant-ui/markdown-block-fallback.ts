// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What one Markdown block should show when it cannot be RENDERED.
 *
 * Streamdown loads two parts of a reply through `React.lazy`: the syntax
 * highlighted code body and the Mermaid renderer. Both are fetched at the
 * moment a reply first contains that construct, and a lazy import that rejects
 * rethrows during render. Nothing between the thread and the router catches
 * that today, so one chunk that will not load replaces the whole application
 * with the router's error page and takes the reply that was already on screen
 * with it.
 *
 * The answer is to keep the CONTENT. A fence whose highlighter is missing is
 * still perfectly readable as text: it is the same characters in the same
 * order, which is what the model actually said. So the degraded form of a block
 * is its own source, with the Markdown fence scaffolding removed so a reader
 * sees code rather than backticks.
 *
 * Deliberately NOT an error card and deliberately not blank. A reader who asked
 * a model for a shell command needs the command, not an apology, and least of
 * all an empty box where their answer used to be.
 */

/** A fence spanning the whole block, the shape Streamdown splits blocks into. */
const WHOLE_BLOCK_FENCE = /^ {0,3}(`{3,}|~{3,})([^\r\n]*)\r?\n([\s\S]*?)(?:\r?\n {0,3}\1\s*)?$/;

export type MarkdownBlockFallback = {
  /** The text to show. Never empty when the block had any content. */
  text: string;
  /** The fence's language tag, when the block was a fence and named one. */
  language: string | null;
  /** Whether the source was a fenced block, so the caller can pick `pre` over prose. */
  fenced: boolean;
};

/**
 * The readable form of a block that failed to render.
 *
 * Falls back to the block's own source unchanged for anything that is not a
 * whole-block fence: a paragraph, a list or a table is already readable as
 * Markdown, and inventing a renderer here would be a second thing to go wrong.
 */
export function markdownBlockFallback(content: string): MarkdownBlockFallback {
  const match = content.match(WHOLE_BLOCK_FENCE);
  if (!match) {
    return { text: content, language: null, fenced: false };
  }
  const language = match[2].trim().split(/\s+/)[0] || null;
  return { text: match[3], language, fenced: true };
}
