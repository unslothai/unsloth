// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fromMarkdown } from "mdast-util-from-markdown";
import { gfmFromMarkdown } from "mdast-util-gfm";
import { mathFromMarkdown } from "mdast-util-math";
import { gfm } from "micromark-extension-gfm";
import { math } from "micromark-extension-math";
import remend from "remend";
import { parseMarkdownIntoBlocks } from "streamdown";

// Cheap gate: a bullet marker then nothing but thematic-break punctuation. The
// ambiguity is not asterisk specific, `- --verbose` and `* ___under___` stream
// through a break frame too. Parsing below decides whether one really renders.
const AMBIGUOUS_BREAK_ITEM_RE = /^[ \t]*([*+-])[ \t]+[*\-_][*\-_ \t]*$/;
const BLOCKQUOTE_PREFIX_RE = /^(?:[ \t]*>[ \t]?)+/;

type MarkdownNode = {
  readonly type: string;
  readonly position?: {
    readonly start: { readonly offset?: number };
    readonly end: { readonly offset?: number };
  };
  readonly children?: readonly MarkdownNode[];
};

// Streamdown parses with GFM plus the math plugin, so a bare CommonMark parse
// disagrees with what is on screen for footnotes and for dollar signs.
function parse(text: string): MarkdownNode {
  return fromMarkdown(text, {
    extensions: [gfm(), math({ singleDollarTextMath: true })],
    mdastExtensions: [gfmFromMarkdown(), mathFromMarkdown()],
  }) as MarkdownNode;
}

// Is a rule on screen right now? Streamdown repairs incomplete Markdown with
// remend and splits into blocks before rendering, so run both: after an
// unclosed construct the repair alone turns this frame into a list already.
// Splitting also keeps the parse to the trailing block rather than the response.
function rendersTrailingThematicBreak(text: string): boolean {
  const block = parseMarkdownIntoBlocks(remend(text)).at(-1);
  if (block === undefined) {
    return false;
  }
  let node = parse(block);
  while (node.children?.length) {
    node = node.children[node.children.length - 1];
  }
  return node.type === "thematicBreak";
}

// Does this marker open an item that will hold text? Matching the offset is
// what separates a nested item continuing a list from `* * *`, a break whose
// completed form is nested lists with no paragraph of its own.
function completesAsTrailingParagraphListItem(
  text: string,
  markerIndex: number,
): boolean {
  const completedText = `${text}x`;
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
  if (!isStreaming) {
    return text;
  }

  const lineStart =
    Math.max(text.lastIndexOf("\n"), text.lastIndexOf("\r")) + 1;
  const line = text.slice(lineStart);
  const blockquotePrefix = line.match(BLOCKQUOTE_PREFIX_RE)?.[0] ?? "";
  const content = line.slice(blockquotePrefix.length);
  const marker = content.match(AMBIGUOUS_BREAK_ITEM_RE)?.[1];
  if (!marker) {
    return text;
  }

  const markerIndex =
    lineStart + blockquotePrefix.length + content.indexOf(marker);
  if (
    !rendersTrailingThematicBreak(text) ||
    !completesAsTrailingParagraphListItem(text, markerIndex)
  ) {
    return text;
  }

  // The line is both a valid thematic break and the streaming prefix of a list
  // item. Buffer it until content arrives instead of rendering the wrong block.
  return text.slice(0, lineStart);
}
