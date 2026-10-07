// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Nodes, Root, RootContent } from "mdast";
import { fromMarkdown } from "mdast-util-from-markdown";
import { gfmFromMarkdown } from "mdast-util-gfm";
import { mathFromMarkdown } from "mdast-util-math";
import { gfm } from "micromark-extension-gfm";
import { math } from "micromark-extension-math";
import { normalizeEscapedInlineMath } from "../../../lib/escaped-inline-math.ts";
import { preprocessLaTeX } from "../../../lib/latex.ts";

/** The words a markdown reply shows, for read-aloud: voices speak raw markup ("asterisk asterisk", #12547). */
export function markdownToSpeechText(markdown: string): string {
  let root: Root;
  try {
    // Same preprocessing as the chat renderer, else "$5 and $10" parses as math and loses its "$".
    root = fromMarkdown(preprocessLaTeX(normalizeEscapedInlineMath(markdown)), {
      extensions: [gfm(), math({ singleDollarTextMath: true })],
      mdastExtensions: [gfmFromMarkdown(), mathFromMarkdown()],
    });
  } catch {
    return markdown;
  }
  const out: string[] = [];
  for (const node of root.children) collectBlocks(node, out, false);
  return out.filter((block) => block.length > 0).join("\n");
}

function collectBlocks(
  node: RootContent,
  out: string[],
  inList: boolean,
): void {
  switch (node.type) {
    case "paragraph":
      out.push(inList ? sentence(inline(node)) : inline(node).trim());
      return;
    // Full stop = the pause the layout gives the eye; else lines run together when spoken.
    case "heading":
      out.push(sentence(inline(node)));
      return;
    case "code":
    case "math":
      out.push(node.value);
      return;
    case "table":
      for (const row of node.children) {
        out.push(sentence(row.children.map(inline).join(", ")));
      }
      return;
    case "html":
    case "thematicBreak":
    case "definition":
    case "footnoteDefinition":
    case "yaml":
      return;
    case "listItem":
      for (const child of node.children) collectBlocks(child, out, true);
      return;
    default:
      if ("children" in node) {
        for (const child of node.children) {
          collectBlocks(child as RootContent, out, inList);
        }
      }
  }
}

function inline(node: Nodes): string {
  switch (node.type) {
    case "text":
    case "inlineCode":
    case "inlineMath":
      return node.value;
    case "break":
      return "\n";
    case "image":
    case "imageReference":
      return node.alt ?? "";
    case "html":
    case "footnoteReference":
      return "";
    default:
      return "children" in node ? node.children.map(inline).join("") : "";
  }
}

function sentence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed || /[.!?:;,…]$/.test(trimmed)) return trimmed;
  return `${trimmed}.`;
}
