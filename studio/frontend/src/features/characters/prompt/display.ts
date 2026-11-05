// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// HTML a reply may legitimately use; any other tag on a line of its own is a card's private markup.
const HTML_TAGS = new Set([
  "a", "b", "blockquote", "br", "code", "details", "div", "em", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img",
  "kbd", "li", "mark", "ol", "p", "pre", "q", "s", "small", "span", "strong", "sub", "summary", "sup", "table", "tbody",
  "td", "th", "thead", "tr", "u", "ul",
]);

/**
 * Cards often wrap blocks in their own tags, such as `<stats>` around a fenced status box. On a line
 * of its own such a tag starts a raw HTML block in Markdown and swallows what follows, so the
 * fence never renders. For display only, those wrapper lines are dropped; the stored text keeps them.
 */
export function tidyRoleplayDisplay(text: string): string {
  if (!text.includes("<")) return text;
  let fence: string | null = null;
  return text
    .split("\n")
    .map((line) => {
      const marker = /^[ \t]*(`{3,}|~{3,})/.exec(line)?.[1];
      if (marker) {
        if (fence === null) fence = marker[0];
        else if (marker[0] === fence) fence = null;
        return line;
      }
      // Code keeps every line, tags included.
      if (fence !== null) return line;
      // (?![\w-]) keeps [^<>]* from re-reading the name, which made a long unclosed "<name" quadratic.
      const tag = /^[ \t]*<\/?([a-z][\w-]*)(?![\w-])[^<>]*>[ \t]*$/i.exec(line)?.[1];
      return tag && !HTML_TAGS.has(tag.toLowerCase()) ? "" : line;
    })
    .join("\n");
}
