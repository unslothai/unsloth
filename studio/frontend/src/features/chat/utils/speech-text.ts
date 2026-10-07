// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { decodeNamedCharacterReference } from "decode-named-character-reference";
import type { Nodes, Root, RootContent } from "mdast";
import { fromMarkdown } from "mdast-util-from-markdown";
import { gfmFromMarkdown } from "mdast-util-gfm";
import { mathFromMarkdown } from "mdast-util-math";
import { gfm } from "micromark-extension-gfm";
import { math } from "micromark-extension-math";
import { defaultRehypePlugins } from "streamdown";
import { normalizeEscapedInlineMath } from "../../../lib/escaped-inline-math.ts";
import { preprocessLaTeX } from "../../../lib/latex.ts";

// Tags the chat renders; any other `<tag>` (`Vec<T>`, `<script>`) stays on the page as text.
const SCHEMA_TAGS = new Set(
  (defaultRehypePlugins.sanitize as [unknown, { tagNames?: string[] }])[1]
    .tagNames,
);

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
  const notes: string[] = [];
  for (const node of root.children) {
    // Footnotes render after the reply, so they are read there too.
    collectBlocks(
      node,
      node.type === "footnoteDefinition" ? notes : out,
      false,
    );
  }
  return [...out, ...notes].filter((block) => block.length > 0).join("\n");
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
    // The sanitizer unwraps raw HTML, so its text shows on the page.
    case "html":
      out.push(htmlText(node.value));
      return;
    case "thematicBreak":
    case "definition":
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
      return htmlText(node.value);
    case "footnoteReference":
      return "";
    default:
      return "children" in node ? node.children.map(inline).join("") : "";
  }
}

function htmlText(html: string): string {
  return html
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(
      /<\/?([a-z][^\s/<>]*)(?:[^<>"']|"[^"]*"|'[^']*')*>/gi,
      (tag, name: string) => (SCHEMA_TAGS.has(name.toLowerCase()) ? " " : tag),
    )
    .replace(/&(#x[\da-f]+|#\d+|[a-z][a-z\d]*);/gi, (ref, body: string) => {
      if (body[0] !== "#") return decodeNamedCharacterReference(body) || ref;
      const hex = body[1] === "x" || body[1] === "X";
      const code = Number.parseInt(body.slice(hex ? 2 : 1), hex ? 16 : 10);
      return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : ref;
    })
    .split("\n")
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .join("\n");
}

function sentence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed || /[.!?:;,…]$/.test(trimmed)) return trimmed;
  return `${trimmed}.`;
}
