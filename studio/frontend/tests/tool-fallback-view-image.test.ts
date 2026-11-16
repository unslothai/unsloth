// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";

import { readSrc } from "./helpers/kit.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

const PATH = "components/assistant-ui/tool-fallback.tsx";
const source = ts.createSourceFile(
  PATH,
  readSrc(PATH),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function findDeclaration(name: string): ts.VariableDeclaration {
  let found: ts.VariableDeclaration | undefined;
  (function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === name)
      found = node;
    if (!found) ts.forEachChild(node, visit);
  })(source);
  assert.ok(found, `${PATH}: no ${name}`);
  return found;
}

function insideContent(node: ts.Node): boolean {
  for (let parent = node.parent; parent; parent = parent.parent) {
    if (
      ts.isJsxElement(parent) &&
      parent.openingElement.tagName.getText(source) === "ToolFallbackContent"
    ) {
      return true;
    }
  }
  return false;
}

test("the view_image picture renders outside the collapsible pane", () => {
  const impl = findDeclaration("ToolFallbackImpl");
  const images: ts.Node[] = [];
  (function visit(node: ts.Node) {
    if (openingTag(node)?.tagName.getText(source) === "ToolFallbackImages")
      images.push(node);
    ts.forEachChild(node, visit);
  })(impl);
  assert.equal(images.length, 1, "the card renders the picture once");
  assert.equal(insideContent(images[0]), false);
});

test("only a finished view_image result moves out of the pane", () => {
  const picture = findDeclaration("picture").initializer?.getText(source) ?? "";
  assert.match(picture, /!isCancelled/);
  assert.match(picture, /toolName === "view_image"/);
  assert.match(picture, /isMcpImageResult\(shown\)/);
});

test("the pane shows only the text of a picture that moved out", () => {
  assert.match(
    source.text,
    /<ToolFallbackResult result=\{picture \? picture\.text : shown\} \/>/,
  );
});
