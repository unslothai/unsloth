// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// HTML a reply may legitimately use; any other tag on a line of its own is a card's private markup.
const HTML_TAGS = new Set([
  "a", "abbr", "b", "blockquote", "br", "caption", "center", "cite", "code", "dd", "del", "details", "div", "dl", "dt",
  "em", "figcaption", "figure", "font", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "ins", "kbd", "li", "mark",
  "ol", "p", "pre", "q", "s", "samp", "section", "small", "span", "strike", "strong", "sub", "summary", "sup", "table",
  "tbody", "td", "tfoot", "th", "thead", "tr", "u", "ul", "var", "wbr",
]);

// Studio's own markup in a reply: an audio-output model's reply is one <audio-player /> line.
const STUDIO_TAGS = new Set(["audio-player"]);

/** Display only: drops a card's own wrapper-tag lines (<stats>), which Markdown treats as raw HTML that swallows the block. */
export function tidyCharacterDisplay(text: string): string {
  if (!text.includes("<")) return text;
  let fence: string | null = null;
  return text
    .split("\n")
    .map((line) => {
      const marker = /^ {0,3}(`{3,}|~{3,})/.exec(line)?.[1];
      if (marker) {
        if (fence === null) fence = marker;
        // CommonMark: a fence closes only on the same character, at least as long.
        else if (marker[0] === fence[0] && marker.length >= fence.length) fence = null;
        return line;
      }
      // Code keeps every line, tags included; four spaces of indent is code too.
      if (fence !== null || /^( {4}|\t)/.test(line)) return line;
      // (?![\w-]) keeps [^<>]* from re-reading the name, which made a long unclosed "<name" quadratic.
      const tag = /^ {0,3}<\/?([a-z][\w-]*)(?![\w-])[^<>]*>[ \t]*$/i.exec(line)?.[1];
      return tag && !HTML_TAGS.has(tag.toLowerCase()) && !STUDIO_TAGS.has(tag.toLowerCase()) ? "" : line;
    })
    .join("\n");
}
