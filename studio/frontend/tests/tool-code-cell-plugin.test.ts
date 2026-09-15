// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

const TOOL_CODE_CELL_PATH = new URL(
  "../src/components/assistant-ui/tool-code-cell.tsx",
  import.meta.url,
);
const source = readFileSync(TOOL_CODE_CELL_PATH, "utf8");
const toolCodeCell = ts.createSourceFile(
  TOOL_CODE_CELL_PATH.pathname,
  source,
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

test("tool code cells do not import the unbounded @streamdown/code plugin", () => {
  assert.doesNotMatch(source, /from ["']@streamdown\/code["']/);
});

function codePluginCalls(): { atModuleScope: boolean }[] {
  const calls: { atModuleScope: boolean }[] = [];
  const visit = (node: ts.Node, insideFunction: boolean): void => {
    if (
      ts.isCallExpression(node) &&
      node.expression.getText(toolCodeCell) === "createCodePlugin"
    ) {
      calls.push({ atModuleScope: !insideFunction });
    }
    const entersFunction =
      insideFunction ||
      ts.isFunctionDeclaration(node) ||
      ts.isFunctionExpression(node) ||
      ts.isArrowFunction(node) ||
      ts.isMethodDeclaration(node);
    node.forEachChild((child) => visit(child, entersFunction));
  };
  toolCodeCell.forEachChild((node) => visit(node, false));
  return calls;
}

test("the tool code cell builds its code plugin once, outside the component", () => {
  const calls = codePluginCalls();
  assert.equal(
    calls.length,
    1,
    "the tool code cell should build exactly one code plugin",
  );
  assert.equal(
    calls[0].atModuleScope,
    true,
    "the code plugin is built inside a component, so its fence slots are discarded on remount",
  );
});
