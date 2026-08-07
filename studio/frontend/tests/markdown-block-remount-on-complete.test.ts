// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

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

function findJsx(
  scope: ts.Node,
  tagName: string,
): ts.JsxOpeningLikeElement | null {
  let found: ts.JsxOpeningLikeElement | null = null;
  const visit = (node: ts.Node): void => {
    if (
      (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
      node.tagName.getText(source) === tagName &&
      !found
    ) {
      found = node;
    }
    node.forEachChild(visit);
  };
  scope.forEachChild(visit);
  return found;
}

function jsxAttribute(
  opening: ts.JsxOpeningLikeElement,
  name: string,
): ts.JsxAttribute | null {
  for (const attribute of opening.attributes.properties) {
    if (
      ts.isJsxAttribute(attribute) &&
      attribute.name.getText(source) === name
    ) {
      return attribute;
    }
  }
  return null;
}

test("StreamdownBlock reads the animating flag from Streamdown's context", () => {
  const block = findFunction("StreamdownBlock");
  assert.ok(block, "StreamdownBlock is missing");

  const body = block.getText(source);
  assert.match(
    body,
    /useContext\(\s*StreamdownContext\s*\)/,
    "StreamdownBlock must read StreamdownContext to know when the message is still streaming",
  );
  assert.match(
    body,
    /isAnimating/,
    "StreamdownBlock must derive its key from isAnimating",
  );
});

test("the streamed block is re-keyed once the message completes", () => {
  const block = findFunction("StreamdownBlock");
  assert.ok(block, "StreamdownBlock is missing");

  // The last <Block> in the function is the default (non-artifact) return; it is
  // the one that renders ordinary prose, so it is the one that accumulates the
  // per-word animation wrappers while streaming.
  let last: ts.JsxOpeningLikeElement | null = null;
  const visit = (node: ts.Node): void => {
    if (
      (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) &&
      node.tagName.getText(source) === "Block"
    ) {
      last = node;
    }
    node.forEachChild(visit);
  };
  block.forEachChild(visit);
  assert.ok(last, "the default <Block> return is missing");

  const key = jsxAttribute(last, "key");
  assert.ok(
    key,
    "the default <Block> must carry a key so completed blocks re-parse without the animation wrappers",
  );

  // The key is either isAnimating itself or a local declared from it.
  const identifier = (key.initializer?.getText(source) ?? "")
    .replace(/[{}]/g, "")
    .trim();
  assert.ok(identifier, "the key must not be empty");
  assert.match(
    block.getText(source),
    new RegExp(`${identifier}\\s*=\\s*isAnimating`),
    `the key ${identifier} must be derived from isAnimating`,
  );
});
