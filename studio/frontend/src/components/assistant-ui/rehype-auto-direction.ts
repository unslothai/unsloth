// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Root, RootContent } from "hast";
import type { Plugin } from "unified";

// The blocks that hold running text: the paragraphs, headings and lists GitHub marks, plus table
// cells, which hold their text directly. A list takes its direction from its first item, which also
// puts its bullets on that side; items are left to the list, as one English item in a Persian list
// is better read in place than with its bullet on the far edge. Code is not here: a fence stays
// left-to-right whatever its comments are written in.
const AUTO_DIRECTION_TAGS = new Set([
  "p",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "ul",
  "ol",
  "th",
  "td",
]);

/** `dir="auto"` on each text block, so the browser lays out each one from its first strong
 * character: a Persian, Arabic or Hebrew paragraph reads right-to-left, its English neighbour does
 * not. Runs after the sanitizer, and a `dir` the model wrote in raw HTML is kept. */
export const rehypeAutoDirection: Plugin<[], Root> =
  function rehypeAutoDirection() {
    return (tree) => {
      function walk(node: Root | RootContent): void {
        if (
          node.type === "element" &&
          AUTO_DIRECTION_TAGS.has(node.tagName) &&
          node.properties.dir == null
        ) {
          node.properties.dir = "auto";
        }
        if ("children" in node) node.children.forEach(walk);
      }
      walk(tree);
    };
  };
