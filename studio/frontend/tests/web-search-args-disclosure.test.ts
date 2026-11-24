// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// JSX cannot run here, so claims go through the TypeScript AST (#17918).

import assert from "node:assert/strict";
import test from "node:test";

import ts from "@typescript/typescript6";

import { readSrc } from "./helpers/kit.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

const CARD_PATH = "components/assistant-ui/tool-ui-web-search.tsx";
const ARGS_SLOT = "tool-web-search-args";

const source = ts.createSourceFile(
  CARD_PATH,
  readSrc(CARD_PATH),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function walk(node: ts.Node, visit: (node: ts.Node) => void): void {
  visit(node);
  node.forEachChild((child) => walk(child, visit));
}

function slotElement(): ts.JsxElement | undefined {
  let hit: ts.JsxElement | undefined;
  walk(source, (node) => {
    if (!ts.isJsxElement(node)) return;
    const slot = openingTag(node)?.attributes.properties.find(
      (p): p is ts.JsxAttribute =>
        ts.isJsxAttribute(p) && p.name.getText(source) === "data-slot",
    )?.initializer;
    if (slot && ts.isStringLiteral(slot) && slot.text === ARGS_SLOT) hit = node;
  });
  return hit;
}

function runningBranchOf(node: ts.Node): "whenTrue" | "whenFalse" | null {
  for (let cur: ts.Node | undefined = node; cur; cur = cur.parent) {
    const parent = cur.parent;
    if (
      parent &&
      ts.isConditionalExpression(parent) &&
      parent.condition.getText(source) === "isRunning"
    ) {
      return parent.whenTrue === cur ? "whenTrue" : "whenFalse";
    }
  }
  return null;
}

test("finished web search card offers the exact call arguments", () => {
  const el = slotElement();
  assert.ok(el, `no element with data-slot="${ARGS_SLOT}"`);
  assert.equal(el.openingElement.tagName.getText(source), "details");
  assert.equal(runningBranchOf(el), "whenFalse");
  assert.match(el.getText(source), /escapeBidiControls\(argsText\)/);
});

test("the card reads argsText from its props", () => {
  assert.match(
    source.getFullText(),
    /WebSearchToolUIImpl: ToolCallMessagePartComponent = \(\{[^}]*\bargsText\b/,
  );
});
