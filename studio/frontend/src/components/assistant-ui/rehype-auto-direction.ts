// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Element, Root, RootContent } from "hast";
import type { Plugin } from "unified";

const TEXT_BLOCKS = new Set([
  "p",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "td",
  "th",
  "dt",
  "dd",
]);
// dir="auto" skips descendants that set their own dir, so it would resolve these from no text at all.
const CONTAINERS = new Set(["ul", "ol", "blockquote"]);
const BLOCKS = new Set([
  ...TEXT_BLOCKS,
  ...CONTAINERS,
  "li",
  "pre",
  "div",
  "table",
  "hr",
]);
const LETTER = /\p{L}/u;
const RTL_LETTER = /[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/;

function firstLetter(node: Element | RootContent): string | null {
  if (node.type === "text") {
    for (const char of node.value) {
      if (LETTER.test(char)) return char;
    }
    return null;
  }
  if (
    node.type !== "element" ||
    node.tagName === "pre" ||
    node.tagName === "code"
  ) {
    return null;
  }
  for (const child of node.children) {
    const found = firstLetter(child);
    if (found) return found;
  }
  return null;
}

// A tight item has no paragraph to carry dir; the item itself keeps the list's side for its marker.
function wrapTightItemText(item: Element): void {
  let end = 0;
  while (end < item.children.length) {
    const child = item.children[end];
    if (child.type === "element" && BLOCKS.has(child.tagName)) break;
    end += 1;
  }
  const inline = item.children.slice(0, end);
  if (!inline.some((child) => firstLetter(child))) return;
  const wrapper: Element = {
    type: "element",
    tagName: "span",
    properties: { dir: "auto" },
    children: inline as Element["children"],
  };
  item.children.splice(0, end, wrapper);
}

export const rehypeAutoDirection: Plugin<[], Root> =
  function rehypeAutoDirection() {
    return (tree) => {
      function walk(node: Root | RootContent): void {
        if (node.type === "element" && node.tagName === "li") {
          wrapTightItemText(node);
        }
        if (node.type === "element" && node.properties.dir === undefined) {
          if (TEXT_BLOCKS.has(node.tagName)) {
            node.properties.dir = "auto";
          } else if (CONTAINERS.has(node.tagName)) {
            // Explicit ltr too, or an English list nested in an RTL one inherits rtl.
            const letter = firstLetter(node);
            if (letter) {
              node.properties.dir = RTL_LETTER.test(letter) ? "rtl" : "ltr";
            }
          }
        }
        if ("children" in node) {
          node.children.forEach(walk);
        }
      }
      walk(tree);
    };
  };
