// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fromMarkdown } from "mdast-util-from-markdown";
import { gfmFromMarkdown } from "mdast-util-gfm";
import { mathFromMarkdown } from "mdast-util-math";
import { gfm } from "micromark-extension-gfm";
import { math } from "micromark-extension-math";
import remend from "remend";
import { parseMarkdownIntoBlocks } from "../../lib/parse-markdown-blocks.ts";

// Cheap gate: bullet marker then only thematic-break punctuation (not asterisk specific).
const AMBIGUOUS_BREAK_ITEM_RE = /^[ \t]*([*+-])[ \t]+[*\-_][*\-_ \t]*$/;
const BLOCKQUOTE_PREFIX_RE = /^(?:[ \t]*>[ \t]?)+/;
// Parsing a long run is quadratic, so a runaway line is left alone.
const MAX_AMBIGUOUS_LINE = 120;

type MarkdownNode = {
  readonly type: string;
  readonly position?: {
    readonly start: { readonly offset?: number };
    readonly end: { readonly offset?: number };
  };
  readonly children?: readonly MarkdownNode[];
};

// Match Streamdown (GFM + math); plain CommonMark differs on footnotes and dollar signs.
function parse(text: string): MarkdownNode {
  return fromMarkdown(text, {
    extensions: [gfm(), math({ singleDollarTextMath: true })],
    mdastExtensions: [gfmFromMarkdown(), mathFromMarkdown()],
  }) as MarkdownNode;
}

function ambiguousMarkerIndex(text: string): number {
  const lineStart =
    Math.max(text.lastIndexOf("\n"), text.lastIndexOf("\r")) + 1;
  const line = text.slice(lineStart);
  if (line.length > MAX_AMBIGUOUS_LINE) {
    return -1;
  }
  const blockquotePrefix = line.match(BLOCKQUOTE_PREFIX_RE)?.[0] ?? "";
  const content = line.slice(blockquotePrefix.length);
  const marker = content.match(AMBIGUOUS_BREAK_ITEM_RE)?.[1];
  return marker === undefined
    ? -1
    : lineStart + blockquotePrefix.length + content.indexOf(marker);
}

// After an unclosed construct the repair alone turns this frame into a list, so test repaired text.
function rendersTrailingThematicBreak(block: string): boolean {
  let node = parse(block);
  while (node.children?.length) {
    node = node.children[node.children.length - 1];
  }
  return node.type === "thematicBreak";
}

// The offset match separates a nested item from `* * *` (nested lists with no paragraph).
function completesAsTrailingParagraphListItem(
  block: string,
  markerIndex: number,
): boolean {
  const completedText = `${block}x`;
  const pending: MarkdownNode[] = [parse(completedText)];
  while (pending.length > 0) {
    const node = pending.pop();
    if (!node) {
      continue;
    }
    if (
      node.type === "listItem" &&
      node.position?.start.offset === markerIndex &&
      node.position.end.offset === completedText.length &&
      node.children?.some(
        (child) =>
          child.type === "paragraph" &&
          child.position?.end.offset === completedText.length,
      )
    ) {
      return true;
    }
    if (node.children) {
      pending.push(...node.children);
    }
  }
  return false;
}

export function stabilizeStreamingMarkdown(
  text: string,
  isStreaming: boolean,
): string {
  if (!isStreaming || ambiguousMarkerIndex(text) < 0) {
    return text;
  }

  // Run what Streamdown runs, reading only the trailing block so cost does not grow with length.
  const block = parseMarkdownIntoBlocks(remend(text)).at(-1);
  const markerIndex = block === undefined ? -1 : ambiguousMarkerIndex(block);
  if (
    block === undefined ||
    markerIndex < 0 ||
    !rendersTrailingThematicBreak(block) ||
    !completesAsTrailingParagraphListItem(block, markerIndex)
  ) {
    return text;
  }

  // Both a valid thematic break and a list-item prefix: hold the line until content arrives.
  return text.slice(
    0,
    Math.max(text.lastIndexOf("\n"), text.lastIndexOf("\r")) + 1,
  );
}
