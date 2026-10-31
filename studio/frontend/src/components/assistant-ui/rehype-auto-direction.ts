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

/** Each text block takes its direction from its own first letter, so Persian, Arabic or Hebrew replies read right to left. */
export const rehypeAutoDirection: Plugin<[], Root> =
  function rehypeAutoDirection() {
    return (tree) => {
      function walk(node: Root | RootContent): void {
        if (node.type === "element" && node.properties.dir === undefined) {
          if (TEXT_BLOCKS.has(node.tagName)) {
            node.properties.dir = "auto";
          } else if (CONTAINERS.has(node.tagName)) {
            // Lists carry the marker padding, so they flip with their text.
            const letter = firstLetter(node);
            if (letter && RTL_LETTER.test(letter)) {
              node.properties.dir = "rtl";
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
