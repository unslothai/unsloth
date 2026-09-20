// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// generation-tool-recovery-approvals.test.ts holds the rule that a call parked before the tab
// closed gets its Approve/Deny back. This holds the BOUND on that rule, at the one place it can be
// expressed: the run has to still be answerable.
//
// A run that settled while it was parked (a backend restart terminalises surviving rows as
// interrupted) has no in-memory _pending slot left. The saved assistant message still carries the
// unresolved card, and no tool_end is ever coming to disarm it, so arming from the seed puts
// Approve/Deny in front of the user whose confirm can only ever return 404. Buttons that cannot
// work are worse than the missing buttons this feature exists to restore.
//
// The guard lives in the follow loop, which no behavioural test reaches without a live EventSource,
// so the call site is pinned here rather than trusted: deleting the guard leaves every other test
// green.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const SOURCE = fileURLToPath(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
);

const parsed = ts.createSourceFile(
  "runtime-provider.tsx",
  readFileSync(SOURCE, "utf8"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

/** Every call to `<something>.armSeededApprovals(...)` in the file. */
const seededApprovalCalls = (): ts.CallExpression[] => {
  const found: ts.CallExpression[] = [];
  const walk = (node: ts.Node): void => {
    if (
      ts.isCallExpression(node) &&
      ts.isPropertyAccessExpression(node.expression) &&
      node.expression.name.text === "armSeededApprovals"
    ) {
      found.push(node);
    }
    ts.forEachChild(node, walk);
  };
  walk(parsed);
  return found;
};

/** Walk up from a node collecting the text of every enclosing `if` condition. */
const enclosingConditions = (node: ts.Node): string[] => {
  const conditions: string[] = [];
  let current: ts.Node | undefined = node.parent;
  while (current) {
    if (ts.isIfStatement(current)) {
      conditions.push(current.expression.getText(parsed));
    }
    current = current.parent;
  }
  return conditions;
};

test("the seeded approvals call exists at all", () => {
  // If this fails the test below would pass vacuously, which is the failure mode that makes a
  // call-site test worthless.
  assert.ok(
    seededApprovalCalls().length > 0,
    "no armSeededApprovals call site found in runtime-provider.tsx",
  );
});

test("seeded approvals are armed only while the run is non-terminal", () => {
  for (const call of seededApprovalCalls()) {
    const conditions = enclosingConditions(call);
    const guarded = conditions.some(
      (text) =>
        text.includes("isTerminalChatGenerationRun") && text.trimStart().startsWith("!"),
    );
    assert.ok(
      guarded,
      "armSeededApprovals must sit under `if (!isTerminalChatGenerationRun(...))`. " +
        "A run that settled while parked has no pending slot left, so the card it arms " +
        "offers buttons that can only 404. Enclosing conditions were: " +
        JSON.stringify(conditions),
    );
  }
});

test("the terminal check is imported, not invented locally", () => {
  const text = readFileSync(SOURCE, "utf8");
  assert.match(
    text,
    /import\s*\{[^}]*isTerminalChatGenerationRun[^}]*\}\s*from/s,
    "isTerminalChatGenerationRun must come from chat-generation-api, so the terminal set " +
      "stays defined in exactly one place",
  );
});
