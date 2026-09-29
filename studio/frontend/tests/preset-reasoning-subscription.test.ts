// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";

const source = ts.createSourceFile(
  "sheet.tsx",
  readSrc("features/chat/chat-settings-sheet.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
const selectors = new Map<
  string,
  (state: Record<string, unknown>) => unknown
>();
function visit(node: ts.Node): void {
  if (
    ts.isVariableDeclaration(node) &&
    ts.isIdentifier(node.name) &&
    ["reasoningBudget", "reasoningBudgetMessage"].includes(node.name.text) &&
    node.initializer &&
    ts.isCallExpression(node.initializer)
  ) {
    selectors.set(
      node.name.text,
      new Function(
        `return (${node.initializer.arguments[0].getText(source)});`,
      )(),
    );
  }
  ts.forEachChild(node, visit);
}
visit(source);

for (const [field, effective, inherited] of [
  ["reasoningBudget", 32, -1],
  ["reasoningBudgetMessage", "Conclude now.", ""],
] as const) {
  test(`preset subscription sees requested-only changes to ${field}`, () => {
    const select = selectors.get(field)!;
    const loaded = `loaded${field[0].toUpperCase()}${field.slice(1)}`;
    const explicit = {
      [field]: effective,
      [loaded]: effective,
      [loaded + "Requested"]: effective,
    };
    const changed = { ...explicit, [loaded + "Requested"]: inherited };
    assert.equal(select(explicit), effective);
    assert.equal(select(changed), inherited);
    assert.notEqual(select(explicit), select(changed));
  });
}
