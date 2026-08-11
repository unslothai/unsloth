// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fromMarkdown } from "mdast-util-from-markdown";
import { isWithinMathBlock } from "remend";

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

function someNode(
  root: MarkdownNode,
  match: (node: MarkdownNode) => boolean,
): boolean {
  const pending: MarkdownNode[] = [root];
  while (pending.length > 0) {
    const node = pending.pop();
    if (!node) {
      continue;
    }
    if (match(node)) {
      return true;
    }
    if (node.children) {
      pending.push(...node.children);
    }
  }
  return false;
}

// The frame really ends in a rule, top level (`* **`) or nested in the bullet
// it opened (`- ___`). Without this a line rendering fine could be hidden.
function rendersTrailingThematicBreak(text: string): boolean {
  return someNode(
    fromMarkdown(text) as MarkdownNode,
    (node) =>
      node.type === "thematicBreak" &&
      node.position?.end.offset === text.length,
  );
}

// The completed form validates the active container while also rejecting
// code, raw HTML, and footnote content.
function completesAsTrailingParagraphListItem(
  text: string,
  markerIndex: number,
): boolean {
  const completedText = `${text}x`;
  return someNode(
    fromMarkdown(completedText) as MarkdownNode,
    (node) =>
      node.type === "listItem" &&
      node.position?.start.offset === markerIndex &&
      node.position.end.offset === completedText.length &&
      (node.children?.some(
        (child) =>
          child.type === "paragraph" &&
          child.position?.end.offset === completedText.length,
      ) ??
        false),
  );
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
    isWithinMathBlock(text, markerIndex) ||
    !rendersTrailingThematicBreak(text) ||
    !completesAsTrailingParagraphListItem(text, markerIndex)
  ) {
    return text;
  }

  // The line is both a valid thematic break and the streaming prefix of a list
  // item. Buffer it until content arrives instead of rendering the wrong block.
  return text.slice(0, lineStart);
}
