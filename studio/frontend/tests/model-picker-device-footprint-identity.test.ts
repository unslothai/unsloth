// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "@typescript/typescript6";

// GgufVariantExpander refetches companion download sizes when resolveDownloadFootprint changes
// identity. The Images and Video pages rebuild that resolver when the text encoder, VAE, memory
// mode or GPU change, so every expander must receive the page's resolver itself, never a stable
// forwarder that hides the change.
const path = "../src/features/model-picker/components/model-selector/pickers.tsx";
const source = ts.createSourceFile(
  path,
  readFileSync(new URL(path, import.meta.url), "utf8"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function visit(node: ts.Node, fn: (node: ts.Node) => void) {
  fn(node);
  node.forEachChild((child) => visit(child, fn));
}

test("every GgufVariantExpander gets the resolver itself, not a stable forwarder", () => {
  const passed: string[] = [];
  visit(source, (node) => {
    if (
      (ts.isJsxSelfClosingElement(node) || ts.isJsxOpeningElement(node)) &&
      node.tagName.getText(source) === "GgufVariantExpander"
    ) {
      for (const attr of node.attributes.properties) {
        if (ts.isJsxAttribute(attr) && attr.name.getText(source) === "resolveDownloadFootprint") {
          const init = attr.initializer;
          assert.ok(init && ts.isJsxExpression(init) && init.expression);
          passed.push(init.expression.getText(source));
        }
      }
    }
  });
  assert.ok(passed.length >= 3, `expected the expander call sites, found ${passed.length}`);
  for (const expression of passed) {
    assert.match(expression, /^(expander\.)?resolveDownloadFootprint$/, expression);
  }
});

test("the On Device expander props memo re-keys on the resolver", () => {
  let found = false;
  visit(source, (node) => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "deviceExpanderProps" &&
      node.initializer &&
      ts.isCallExpression(node.initializer)
    ) {
      found = true;
      const deps = node.initializer.arguments[1];
      assert.ok(deps && ts.isArrayLiteralExpression(deps));
      assert.ok(
        deps.elements.some((e) => e.getText(source) === "resolveDownloadFootprint"),
        deps.getText(source),
      );
    }
  });
  assert.ok(found, "deviceExpanderProps");
});
