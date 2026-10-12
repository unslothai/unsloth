// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Tools section was hidden for external models while Unsloth's tool loop still applied
// its 25-call budget to them (#13243). The section is too large to render here, so its JSX is
// walked with isExternalModel / the capability gate fixed to each side.

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import { readSrc } from "./helpers/kit.ts";

const SHEET = ts.createSourceFile(
  "chat-settings-sheet.tsx",
  readSrc("features/chat/chat-settings-sheet.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

type Gates = { isExternalModel: boolean; showStudioToolLoopControls: boolean };

function find<T extends ts.Node>(
  root: ts.Node,
  pick: (node: ts.Node) => node is T,
): T | null {
  let found: T | null = null;
  const visit = (node: ts.Node): void => {
    if (found) return;
    if (pick(node)) {
      found = node;
      return;
    }
    ts.forEachChild(node, visit);
  };
  visit(root);
  return found;
}

function toolsSection(): ts.JsxElement {
  const section = find(
    SHEET,
    (node): node is ts.JsxElement =>
      ts.isJsxElement(node) &&
      node.openingElement.tagName.getText() === "CollapsibleSection" &&
      node.openingElement.attributes.properties.some(
        (a) =>
          ts.isJsxAttribute(a) &&
          a.name.getText() === "label" &&
          a.initializer?.getText() === '"Tools"',
      ),
  );
  assert.ok(section, "Tools section not found");
  return section;
}

function evalGate(expr: ts.Expression, gates: Gates): boolean {
  const name = expr.getText();
  if (name in gates) return gates[name as keyof Gates];
  if (
    ts.isPrefixUnaryExpression(expr) &&
    expr.operator === ts.SyntaxKind.ExclamationToken
  ) {
    return !evalGate(expr.operand, gates);
  }
  throw new Error(`unexpected gate in Tools section: ${name}`);
}

// Component names rendered under the given gates, following ternaries around and inside the section.
function rendered(gates: Gates): string[] {
  const out: string[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isConditionalExpression(node)) {
      visit(evalGate(node.condition, gates) ? node.whenTrue : node.whenFalse);
      return;
    }
    if (ts.isJsxSelfClosingElement(node)) out.push(node.tagName.getText());
    ts.forEachChild(node, visit);
  };
  let outer: ts.Node = toolsSection();
  while (
    outer.parent &&
    (ts.isParenthesizedExpression(outer.parent) ||
      ts.isConditionalExpression(outer.parent))
  ) {
    outer = outer.parent;
  }
  if (ts.isConditionalExpression(outer)) visit(outer);
  else visit(toolsSection());
  return out.filter((name) => /Toggle$|Slider$/.test(name));
}

test("external models with Unsloth tools see the tool loop's budget, duration and dedupe", () => {
  const external = rendered({
    isExternalModel: true,
    showStudioToolLoopControls: true,
  });
  for (const control of [
    "MaxToolCallsSlider",
    "ToolCallTimeoutSlider",
    "DeduplicateToolCallsToggle",
  ]) {
    assert.ok(external.includes(control), `${control} hidden for external`);
  }
  // Nudge is forced off on external turns, so its toggle would do nothing there.
  assert.ok(!external.includes("NudgeToolCallsToggle"));
});

test("external providers without Unsloth tools get no Tools section", () => {
  assert.deepEqual(
    rendered({ isExternalModel: true, showStudioToolLoopControls: false }),
    [],
  );
});

test("local models keep every tool control", () => {
  assert.deepEqual(
    rendered({ isExternalModel: false, showStudioToolLoopControls: true }),
    [
      "AutoHealToolCallsToggle",
      "NudgeToolCallsToggle",
      "DeduplicateToolCallsToggle",
      "ConfirmToolCallsToggle",
      "BypassPermissionsToggle",
      "MaxToolCallsSlider",
      "ToolCallTimeoutSlider",
    ],
  );
});

test("the gate matches the external request's own Unsloth-tools test", () => {
  const sheet = SHEET.getFullText();
  assert.match(
    sheet,
    /showStudioToolLoopControls =\s*!isExternalModel \|\|\s*providerModelSupportsStudioTools\([^)]*\) === true;/,
  );
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /externalUsesStudioTools =\s*providerModelSupportsStudioTools\([^)]*\) === true;/,
  );
  // Every knob shown for external is one the external tool-loop body sends.
  const loopBody = adapter.slice(
    adapter.indexOf("run_tools_locally: true") - 2500,
    adapter.indexOf("run_tools_locally: true"),
  );
  for (const field of [
    "max_tool_calls_per_message: runtime.maxToolCallsPerMessage",
    "tool_call_timeout:",
    "deduplicate_tool_calls: runtime.deduplicateToolCalls",
  ]) {
    assert.ok(loopBody.includes(field), `external body lacks ${field}`);
  }
});
