// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Walks the AST: a text check cannot tell Promise.all from Promise.race, and race reopens
// the row-write race. Kept separate so only tests needing the TS compiler load it.

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import { readSrc } from "./helpers/kit.ts";

const source = ts.createSourceFile(
  "runtime-provider.tsx",
  readSrc("features/chat/runtime-provider.tsx"),
  ts.ScriptTarget.ES2022,
  true,
  ts.ScriptKind.TSX,
);

function collect(root: ts.Node, match: (node: ts.Node) => boolean): ts.Node[] {
  const found: ts.Node[] = [];
  const walk = (node: ts.Node): void => {
    if (match(node)) found.push(node);
    ts.forEachChild(node, walk);
  };
  walk(root);
  return found;
}

function isCombinator(node: ts.Node, name: string): node is ts.CallExpression {
  if (!ts.isCallExpression(node)) return false;
  const callee = node.expression;
  return (
    ts.isPropertyAccessExpression(callee) &&
    ts.isIdentifier(callee.expression) &&
    callee.expression.text === "Promise" &&
    callee.name.text === name
  );
}

function calls(root: ts.Node, name: string): ts.Node[] {
  return collect(
    root,
    (node) =>
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === name,
  );
}

function syncBody(): ts.Node {
  const declarations = collect(
    source,
    (node) =>
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === "sync" &&
      node.initializer !== undefined &&
      ts.isArrowFunction(node.initializer),
  ) as ts.VariableDeclaration[];
  assert.equal(
    declarations.length,
    1,
    "expected exactly one `const sync = () => ...` in runtime-provider",
  );
  return declarations[0].initializer as ts.ArrowFunction;
}

function prerequisites(): ts.CallExpression {
  const gates = collect(syncBody(), (node) => isCombinator(node, "all"));
  assert.equal(
    gates.length,
    1,
    "expected exactly one Promise.all inside sync(): the read's prerequisites",
  );
  return gates[0] as ts.CallExpression;
}

test("the read waits for ALL of its prerequisites, not whichever answers first", () => {
  // Promise.race here would let the GET start while the row POST is still in flight.
  const gate = prerequisites();
  assert.equal(gate.arguments.length, 1, "Promise.all takes one array");
  const [waits] = gate.arguments;
  assert.ok(ts.isArrayLiteralExpression(waits), "Promise.all argument is an array literal");

  for (const wait of [
    "awaitThreadScopedSettingsWrite",
    "awaitStoredChatThreadWrites",
  ]) {
    assert.ok(
      waits.elements.some((element) => calls(element, wait).length > 0),
      `${wait}() is not one of the gated prerequisites`,
    );
  }
});

test("the read hangs off the prerequisites rather than running beside them", () => {
  const gate = prerequisites();
  const parent = gate.parent;
  assert.ok(
    ts.isPropertyAccessExpression(parent) && parent.name.text === "then",
    "the prerequisites are not immediately followed by .then",
  );
  const then = parent.parent;
  assert.ok(ts.isCallExpression(then), "the .then is not called");
  assert.ok(
    calls(then.arguments[0], "getStoredChatThreadReadResult").length > 0,
    "the thread read does not happen inside the prerequisites' .then",
  );
  assert.equal(
    calls(syncBody(), "getStoredChatThreadReadResult").length,
    1,
    "more than one thread read in sync(); only the gated one may exist",
  );
});

test("the whole attempt, waits included, sits inside one deadline", () => {
  // The deadline must CONTAIN the prerequisites; neither wait is bounded on its own.
  const gate = prerequisites();
  const races = collect(syncBody(), (node) => isCombinator(node, "race"));
  assert.ok(races.length >= 1, "the per-attempt deadline is gone");

  const enclosing = races.find(
    (race) => race.getStart() <= gate.getStart() && race.getEnd() >= gate.getEnd(),
  );
  assert.ok(enclosing, "the prerequisites are not inside a Promise.race deadline");

  const [candidates] = (enclosing as ts.CallExpression).arguments;
  assert.ok(ts.isArrayLiteralExpression(candidates));
  assert.ok(
    candidates.elements.some(
      (element) =>
        element !== gate &&
        element.getText().includes("THREAD_READ_TIMEOUT_MS") &&
        /reject\(/.test(element.getText()),
    ),
    "the deadline does not reject on THREAD_READ_TIMEOUT_MS; a deadline that RESOLVES " +
      "would fall through to the read, find no row, and release this chat's held edits " +
      "into the installation defaults",
  );
});
