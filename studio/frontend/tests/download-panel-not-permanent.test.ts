// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Downloads overlay must unmount when the job list is empty (#9849, which
 * shipped a permanent corner FAB and was reverted by #10298).
 *
 * The second consequence is the one that reads as harmless: the rail is a
 * `flex flex-col gap-2` column, so a panel returning a wrapper instead of
 * `null` takes a slot and its gap, pushing the loaded models card off the
 * corner it is meant to hold.
 *
 * Read from the source: the node suite has no DOM. Matched through the AST, so
 * reformatting or a rename of `jobKeys` cannot quietly retire it.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import { openingTag } from "./helpers/tsx-ast.ts";

const parse = (relative: string, name: string): ts.SourceFile =>
  ts.createSourceFile(
    name,
    readFileSync(new URL(`../src/${relative}`, import.meta.url), "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );

const PANEL_PATH = "features/hub/download-manager/download-manager-panel.tsx";
const panel = parse(PANEL_PATH, "download-manager-panel.tsx");
const provider = parse("app/provider.tsx", "provider.tsx");

function panelComponent(): ts.FunctionDeclaration {
  let found: ts.FunctionDeclaration | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isFunctionDeclaration(node) &&
      node.name?.getText() === "DownloadManagerPanel"
    ) {
      found = node;
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(panel, visit);
  assert.ok(found, `${PANEL_PATH} no longer exports a DownloadManagerPanel`);
  return found;
}

/** The ordered-job-keys local, found by its initializer so a rename cannot slip past. */
function jobKeysBinding(component: ts.FunctionDeclaration): string {
  let name: string | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.initializer &&
      ts.isCallExpression(node.initializer) &&
      node.initializer.expression.getText() === "useDownloadManagerStore" &&
      node.initializer.arguments.some((arg) =>
        /orderedJobKeys/i.test(arg.getText()),
      )
    ) {
      name = node.name.getText();
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(component, visit);
  assert.ok(
    name,
    `${PANEL_PATH}: no local is bound from useDownloadManagerStore with the
ordered-job-keys selector, so the guard below cannot be located. If the
selector was renamed, update this test; if the panel stopped reading the
job list at all, it can no longer know when to unmount.`,
  );
  return name;
}

/**
 * Does `condition` test for an empty `jobKeys`? Operand order matters: an
 * unordered match also accepts `0 < jobKeys.length`, the exact inverse, and the
 * always-true `0 <= jobKeys.length`. Hence the explicit shapes.
 */
function testsForAnEmptyList(condition: ts.Node, jobKeys: string): boolean {
  const length = `${jobKeys}.length`;
  const K = ts.SyntaxKind;
  // [left, operator, right], each meaning "the list is empty".
  const shapes: [string, ts.SyntaxKind, string][] = [
    [length, K.EqualsEqualsEqualsToken, "0"],
    ["0", K.EqualsEqualsEqualsToken, length],
    [length, K.LessThanToken, "1"],
    [length, K.LessThanEqualsToken, "0"],
    ["0", K.GreaterThanEqualsToken, length],
    ["1", K.GreaterThanToken, length],
  ];
  let ok = false;
  const visit = (node: ts.Node): void => {
    if (ts.isBinaryExpression(node)) {
      const left = node.left.getText();
      const right = node.right.getText();
      const op = node.operatorToken.kind;
      if (shapes.some(([l, o, r]) => left === l && op === o && right === r)) {
        ok = true;
      }
    }
    if (
      ts.isPrefixUnaryExpression(node) &&
      node.operator === K.ExclamationToken &&
      node.operand.getText() === length
    ) {
      ok = true;
    }
    ts.forEachChild(node, visit);
  };
  visit(condition);
  return ok;
}

function returnsNull(statement: ts.Statement): boolean {
  const body = ts.isBlock(statement) ? statement.statements : [statement];
  return body.some(
    (node) =>
      ts.isReturnStatement(node) &&
      node.expression?.kind === ts.SyntaxKind.NullKeyword,
  );
}

test("the Downloads overlay unmounts when there are no jobs", () => {
  const component = panelComponent();
  const jobKeys = jobKeysBinding(component);
  const statements = component.body?.statements ?? ts.factory.createNodeArray();

  // Top level only: a nested guard still leaves mounted-and-empty states.
  const guarded = statements.some(
    (statement) =>
      ts.isIfStatement(statement) &&
      returnsNull(statement.thenStatement) &&
      testsForAnEmptyList(statement.expression, jobKeys),
  );

  assert.ok(
    guarded,
    `${PANEL_PATH}: DownloadManagerPanel must return null while ${jobKeys} is
empty, at the top level of the component.

Without it the Downloads overlay is permanent: a bottom-right FAB on
every hub route of a fresh install, and a flex slot in the rail that
pushes the loaded models card off the corner even when it paints
nothing. That is PR #9849, which had to be reverted by PR #10298.

Expected something of the shape:  if (... || ${jobKeys}.length === 0) return null;`,
  );
});

test("the rail-facing wrapper can still be squeezed by the rail's cap", () => {
  // min-h-0: without it a flex item floors at auto and the capped rail squeezes
  // the update card above instead of this list.
  const source = readFileSync(
    new URL(`../src/${PANEL_PATH}`, import.meta.url),
    "utf8",
  );
  const ternary = source.match(/positioned\s*\?\s*"([^"]*)"\s*:\s*"([^"]*)"/);
  assert.ok(
    ternary,
    `${PANEL_PATH}: the positioned/stacked className ternary is gone, so the
rail-facing branch can no longer be checked`,
  );
  assert.match(
    ternary[1],
    /\bfixed\b.*\bbottom-4\b.*\bright-4\b/,
    `${PANEL_PATH}: the standalone panel has left the bottom-right corner`,
  );
  assert.match(
    ternary[2],
    /\bmin-h-0\b/,
    `${PANEL_PATH}: the stacked branch dropped min-h-0, so a capped rail will
squeeze the update card above this list instead of the list`,
  );
});

test("the Downloads panel sits above the loaded models card in both rails", () => {
  // The loaded models card holds the corner; this one comes and goes above it.
  const tags: { name: string; at: number }[] = [];
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    const name = opening?.tagName.getText();
    if (name === "DownloadManagerPanel" || name === "LoadedModelsIndicator") {
      tags.push({ name, at: node.getStart() });
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(provider, visit);
  tags.sort((a, b) => a.at - b.at);

  const panels = tags.filter((t) => t.name === "DownloadManagerPanel");
  const indicators = tags.filter((t) => t.name === "LoadedModelsIndicator");
  assert.ok(
    panels.length > 0 && panels.length === indicators.length,
    `provider.tsx: expected the download panel and the loaded models card to be
paired in every rail, found ${panels.length} and ${indicators.length}`,
  );
  for (let i = 0; i < panels.length; i += 1) {
    assert.ok(
      panels[i].at < indicators[i].at,
      "provider.tsx: DownloadManagerPanel is rendered below LoadedModelsIndicator.\n" +
        "The rail is a bottom-anchored column, so the last child is the one on\n" +
        "the corner, and that is meant to be the loaded models card.",
    );
  }
});
