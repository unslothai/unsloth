// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fromMarkdown } from "mdast-util-from-markdown";
import { gfmFromMarkdown } from "mdast-util-gfm";
import { mathFromMarkdown } from "mdast-util-math";
import { gfm } from "micromark-extension-gfm";
import { math } from "micromark-extension-math";
import remend from "remend";
import { parseMarkdownIntoBlocks } from "streamdown";

// Cheap gate: a bullet marker then only thematic-break punctuation. Not
// asterisk specific, `- --verbose` and `* ___under___` flash too.
const AMBIGUOUS_BREAK_ITEM_RE = /^[ \t]*([*+-])[ \t]+[*\-_][*\-_ \t]*$/;
const BLOCKQUOTE_PREFIX_RE = /^(?:[ \t]*>[ \t]?)+/;
// Real prefixes are short, and parsing a long run is quadratic (100k dashes
// cost 8s), so a runaway line is left alone.
const MAX_AMBIGUOUS_LINE = 120;

type MarkdownNode = {
  readonly type: string;
  readonly position?: {
    readonly start: { readonly offset?: number };
    readonly end: { readonly offset?: number };
  };
  readonly children?: readonly MarkdownNode[];
};

// Streamdown parses with GFM and the math plugin; plain CommonMark disagrees
// on footnotes and on dollar signs.
function parse(text: string): MarkdownNode {
  return fromMarkdown(text, {
    extensions: [gfm(), math({ singleDollarTextMath: true })],
    mdastExtensions: [gfmFromMarkdown(), mathFromMarkdown()],
  }) as MarkdownNode;
}

// Index of the trailing line's bullet marker, or -1 if it is not ambiguous.
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

// Is a rule on screen now? After an unclosed construct the repair alone turns
// this frame into a list, so unrepaired text is the wrong thing to test.
function rendersTrailingThematicBreak(block: string): boolean {
  let node = parse(block);
  while (node.children?.length) {
    node = node.children[node.children.length - 1];
  }
  return node.type === "thematicBreak";
}

// Will this marker hold text? The offset match separates a nested item from
// `* * *`, whose completed form is nested lists with no paragraph of its own.
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

  // Run what Streamdown runs: repair, split, then read only the trailing block
  // so the cost does not grow with the response.
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

  // Both a valid thematic break and a list-item prefix: hold the line back
  // until content arrives instead of rendering the wrong block.
  return text.slice(
    0,
    Math.max(text.lastIndexOf("\n"), text.lastIndexOf("\r")) + 1,
  );
}
