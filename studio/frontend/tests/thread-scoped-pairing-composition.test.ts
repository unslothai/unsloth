// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sibling invariants file asserts on source TEXT, which pins which calls are present and
// in what order but not how they COMBINE -- and here the combinator is the invariant. Turning
// the inner `Promise.all` into `Promise.race` is one token, leaves every call present and
// every index unmoved, and reinstates the race the gate closed: the GET then fires as soon as
// EITHER prerequisite settles, so a first send can still overtake the row write, find no row,
// and release the chat's held edits into the installation defaults. Measured on the parent
// commit, that edit passes the whole suite, 4045 of 4045.
//
// So these walk the syntax tree. Kept out of the text-based file like tsx-ast.ts and
// module-stubs.ts: only tests that need the TypeScript compiler should pay to load it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

const source = ts.createSourceFile(
  "runtime-provider.tsx",
  readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  ),
  ts.ScriptTarget.ES2022,
  true,
  ts.ScriptKind.TSX,
);

/** Every node under `root` satisfying `match`, outermost first. */
function collect(root: ts.Node, match: (node: ts.Node) => boolean): ts.Node[] {
  const found: ts.Node[] = [];
  const walk = (node: ts.Node): void => {
    if (match(node)) found.push(node);
    ts.forEachChild(node, walk);
  };
  walk(root);
  return found;
}

/** `Promise.<name>(...)` as a call. */
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

/** Calls to the free function `name` anywhere under `root`. */
function calls(root: ts.Node, name: string): ts.Node[] {
  return collect(
    root,
    (node) =>
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === name,
  );
}

/** The `sync` closure inside ThreadScopedSettingsSync, which is the whole subject here. */
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

/** The Promise.all whose settling is what lets the read start. */
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
  // Promise.race here would let the GET start while the row POST is still in flight. Both
  // waits would still be present, and still textually ahead of the read, so nothing that
  // reads this file as a string can tell the difference.
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
  // Two independent promises would issue the GET immediately, however the waits combine.
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
  // The structural form of the sibling file's index check: the deadline must CONTAIN the
  // prerequisites, not just precede them in the text. Neither wait is bounded on its own, so
  // outside it a stalled write ends in a refused send.
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
