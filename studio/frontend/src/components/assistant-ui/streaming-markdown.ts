// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fromMarkdown } from "mdast-util-from-markdown";
import { isWithinMathBlock } from "remend";

const AMBIGUOUS_BOLD_ASTERISK_ITEM_RE = /^[ \t]*\*[ \t]+\*{2,}[ \t]*$/;
const BLOCKQUOTE_PREFIX_RE = /^(?:[ \t]*>[ \t]?)+/;

type MarkdownNode = {
  readonly type: string;
  readonly position?: {
    readonly start: { readonly offset?: number };
    readonly end: { readonly offset?: number };
  };
  readonly children?: readonly MarkdownNode[];
};

function isTrailingThematicBreak(text: string, markerIndex: number): boolean {
  const pending: MarkdownNode[] = [fromMarkdown(text) as MarkdownNode];
  while (pending.length > 0) {
    const node = pending.pop();
    if (!node) {
      continue;
    }
    if (
      node.type === "thematicBreak" &&
      node.position?.start.offset === markerIndex &&
      node.position.end.offset === text.length
    ) {
      return true;
    }
    if (node.children) {
      pending.push(...node.children);
    }
  }
  return false;
}

function becomesTrailingParagraphListItem(
  text: string,
  markerIndex: number,
): boolean {
  const completedText = `${text}x`;
  const pending: MarkdownNode[] = [fromMarkdown(completedText) as MarkdownNode];
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
  if (!AMBIGUOUS_BOLD_ASTERISK_ITEM_RE.test(content)) {
    return text;
  }

  const markerIndex =
    lineStart + blockquotePrefix.length + content.indexOf("*");
  if (
    isWithinMathBlock(text, markerIndex) ||
    !isTrailingThematicBreak(text, markerIndex) ||
    !becomesTrailingParagraphListItem(text, markerIndex)
  ) {
    return text;
  }

  // `* **` is both a valid thematic break and the streaming prefix of an
  // asterisk list item whose content starts bold. Buffer the ambiguous line
  // until text arrives, rather than briefly rendering the wrong block type.
  return text.slice(0, lineStart);
}
