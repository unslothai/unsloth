// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const MARKDOWN_TEXT_PATH = fileURLToPath(
  new URL("../src/components/assistant-ui/markdown-text.tsx", import.meta.url),
);
const source = ts.createSourceFile(
  MARKDOWN_TEXT_PATH,
  readFileSync(MARKDOWN_TEXT_PATH, "utf8"),
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

function findFunction(name: string): ts.FunctionDeclaration | null {
  let found: ts.FunctionDeclaration | null = null;
  const visit = (node: ts.Node): void => {
    if (ts.isFunctionDeclaration(node) && node.name?.getText(source) === name) {
      found = node;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return found;
}

const ANIMATION_FREE_BLOCK_PROPS_RE = /^<Block\s+\{\.\.\.blockProps\}/;
const ANIMATION_FREE_HELPER_RE = /withoutStreamdownAnimationPlugin/;
const NULL_ANIMATION_PLUGIN_RE = /animatePlugin:\s*null/;
// The block component is now rendered inside MarkdownBlockBoundary, because a
// lazy chunk that fails to load anywhere in a reply used to unmount all of
// Unsloth through the router's error boundary. `memo` is still the OUTERMOST
// wrapper, which is the property this pins: on a memo hit React skips the
// boundary and the block together, exactly as it did before. Pinned to the new
// spelling rather than loosened, because a version with memo INSIDE the boundary
// would render one component per block per token and no output test could tell
// the two apart.
const MEMOISED_BLOCK_RE =
  /const\s+StreamdownBlock\s*=\s*memo\(\s*\(props: BlockProps\) => \(\s*<MarkdownBlockBoundary content=\{props\.content\}>\s*<StreamdownBlockContent \{\.\.\.props\} \/>/;

test("the block props use the animation-free plugin list", () => {
  const helper = findFunction("useAnimationFreeBlockProps");
  assert.ok(helper, "useAnimationFreeBlockProps is missing");

  const body = helper.getText(source);
  assert.match(
    body,
    ANIMATION_FREE_HELPER_RE,
    "the helper must filter Streamdown's animation rehype plugin",
  );
  assert.match(
    body,
    NULL_ANIMATION_PLUGIN_RE,
    "the filtered animation plugin must not reach Streamdown's Block",
  );
});

test("every Markdown block receives the animation-free props", () => {
  const block = findFunction("StreamdownBlockContent");
  assert.ok(block, "StreamdownBlockContent is missing");

  const renderedBlocks: ts.JsxOpeningLikeElement[] = [];
  const visit = (node: ts.Node): void => {
    if (
      (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
      node.tagName.getText(source) === "Block"
    ) {
      renderedBlocks.push(node);
    }
    node.forEachChild(visit);
  };
  block.forEachChild(visit);
  assert.ok(renderedBlocks.length > 0, "StreamdownBlock renders no blocks");
  for (const renderedBlock of renderedBlocks) {
    assert.match(
      renderedBlock.getText(source),
      ANIMATION_FREE_BLOCK_PROPS_RE,
      "every <Block> path must use the animation-free props",
    );
  }
});

test("completed custom blocks are memoised while the tail streams", () => {
  assert.match(source.getText(), MEMOISED_BLOCK_RE);
});
