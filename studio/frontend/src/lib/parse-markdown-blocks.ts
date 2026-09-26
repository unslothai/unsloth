// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
//
// Block splitter aligned with vercel/streamdown#608 (blockTokens only). Vendored from
// streamdown `lib/parse-blocks.tsx` until a published release includes that change.
// Incremental reuse from the same PR is omitted here so splits stay identical to
// streamdown 2.5 while streaming; drop this file once streamdown ships #608.

import { Lexer, type Token } from "marked";

const footnoteReferencePattern = /\[\^[\w-]{1,200}\](?!:)/;
const footnoteDefinitionPattern = /\[\^[\w-]{1,200}\]:/;
const openingTagPattern = /<([A-Za-z][\w:-]*)[\s>/]/;

const voidElements = new Set([
  "area",
  "base",
  "br",
  "col",
  "embed",
  "hr",
  "img",
  "input",
  "link",
  "meta",
  "param",
  "source",
  "track",
  "wbr",
]);

const openTagPatternCache = new Map<string, RegExp>();
const closeTagPatternCache = new Map<string, RegExp>();

const getOpenTagPattern = (tagName: string): RegExp => {
  const normalizedTag = tagName.toLowerCase();
  const cached = openTagPatternCache.get(normalizedTag);
  if (cached) {
    return cached;
  }
  const pattern = new RegExp(`<${normalizedTag}(?=[\\s>/])[^>]*>`, "gi");
  openTagPatternCache.set(normalizedTag, pattern);
  return pattern;
};

const getCloseTagPattern = (tagName: string): RegExp => {
  const normalizedTag = tagName.toLowerCase();
  const cached = closeTagPatternCache.get(normalizedTag);
  if (cached) {
    return cached;
  }
  const pattern = new RegExp(`</${normalizedTag}(?=[\\s>])[^>]*>`, "gi");
  closeTagPatternCache.set(normalizedTag, pattern);
  return pattern;
};

const countNonSelfClosingOpenTags = (
  block: string,
  tagName: string,
): number => {
  if (voidElements.has(tagName.toLowerCase())) {
    return 0;
  }
  const matches = block.match(getOpenTagPattern(tagName));
  if (!matches) {
    return 0;
  }
  let count = 0;
  for (const match of matches) {
    if (!match.trimEnd().endsWith("/>")) {
      count += 1;
    }
  }
  return count;
};

const countClosingTags = (block: string, tagName: string): number => {
  const matches = block.match(getCloseTagPattern(tagName));
  return matches ? matches.length : 0;
};

const countDoubleDollars = (str: string): number => {
  let count = 0;
  for (let i = 0; i < str.length - 1; i += 1) {
    if (str[i] === "$" && str[i + 1] === "$") {
      count += 1;
      i += 1;
    }
  }
  return count;
};

const lineEndingPattern = /\r\n|\r/g;

const lexBlocks = (markdown: string): Token[] =>
  new Lexer({ gfm: true }).blockTokens(markdown);

const mergeTokensIntoBlocks = (tokens: Token[]): string[] => {
  const mergedBlocks: string[] = [];
  const htmlStack: string[] = [];
  let previousTokenWasCode = false;

  for (const token of tokens) {
    const currentBlock = token.raw;
    const mergedBlocksLen = mergedBlocks.length;

    if (htmlStack.length > 0) {
      mergedBlocks[mergedBlocksLen - 1] += currentBlock;

      const trackedTag = htmlStack.at(-1) as string;
      const newOpenTags = countNonSelfClosingOpenTags(currentBlock, trackedTag);
      const newCloseTags = countClosingTags(currentBlock, trackedTag);

      for (let i = 0; i < newOpenTags; i += 1) {
        htmlStack.push(trackedTag);
      }
      for (let i = 0; i < newCloseTags; i += 1) {
        if (htmlStack.length > 0 && htmlStack.at(-1) === trackedTag) {
          htmlStack.pop();
        }
      }
      continue;
    }

    if (token.type === "html" && token.block) {
      const openingTagMatch = currentBlock.match(openingTagPattern);
      if (openingTagMatch) {
        const tagName = openingTagMatch[1];
        const openTags = countNonSelfClosingOpenTags(currentBlock, tagName);
        const closeTags = countClosingTags(currentBlock, tagName);
        if (openTags > closeTags) {
          htmlStack.push(tagName);
        }
      }
    }

    if (mergedBlocksLen > 0 && !previousTokenWasCode) {
      const previousBlock = mergedBlocks[mergedBlocksLen - 1];
      const prevDollarCount = countDoubleDollars(previousBlock);

      if (prevDollarCount % 2 === 1) {
        mergedBlocks[mergedBlocksLen - 1] = previousBlock + currentBlock;
        continue;
      }
    }

    mergedBlocks.push(currentBlock);

    if (token.type !== "space") {
      previousTokenWasCode = token.type === "code";
    }
  }

  return mergedBlocks;
};

export function parseMarkdownIntoBlocks(markdown: string): string[] {
  const hasFootnoteReference = footnoteReferencePattern.test(markdown);
  const hasFootnoteDefinition = footnoteDefinitionPattern.test(markdown);

  if (hasFootnoteReference || hasFootnoteDefinition) {
    return [markdown];
  }

  const input = markdown.includes("\r")
    ? markdown.replace(lineEndingPattern, "\n")
    : markdown;

  return mergeTokensIntoBlocks(lexBlocks(input));
}
