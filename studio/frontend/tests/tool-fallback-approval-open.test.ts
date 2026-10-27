// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A generic tool card (MCP tools, edit_file, ...) parked on Allow / Deny is pinned open, as the
// named cards are, so the arguments being approved are on screen even under "Collapsed".
import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";

import { readSrc } from "./helpers/kit.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

const PATH = "components/assistant-ui/tool-fallback.tsx";
const source = ts.createSourceFile(PATH, readSrc(PATH), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

function findDeclaration(name: string): ts.Node {
  let found: ts.Node | undefined;
  (function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === name) found = node;
    if (!found) ts.forEachChild(node, visit);
  })(source);
  assert.ok(found, `${PATH}: no ${name}`);
  return found;
}

test("the generic card passes its approval state to the card root", () => {
  const impl = findDeclaration("ToolFallbackImpl");
  const initializers = new Map<string, string>();
  let rootAttr: string | undefined;
  (function visit(node: ts.Node) {
    if (ts.isVariableDeclaration(node) && node.initializer) {
      initializers.set(node.name.getText(source), node.initializer.getText(source));
    }
    const tag = openingTag(node);
    if (tag?.tagName.getText(source) === "ToolFallbackRoot") {
      for (const attr of tag.attributes.properties) {
        if (ts.isJsxAttribute(attr) && attr.name.getText(source) === "awaitingApproval") {
          rootAttr = attr.initializer?.getText(source).replace(/^\{|\}$/g, "") ?? "true";
        }
      }
    }
    ts.forEachChild(node, visit);
  })(impl);
  assert.ok(rootAttr, "ToolFallbackRoot gets no awaitingApproval, so the card stays collapsed while it waits");
  assert.equal(initializers.get(rootAttr), "useToolAwaitingApproval(toolCallId)");
});

test("the card root opens for an approval above the collapse preference", () => {
  const root = readSrc(PATH);
  assert.match(root, /const isOpen =\s*awaitingApproval \|\|/);
});
