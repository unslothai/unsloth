// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sibling invariants file asserts on source TEXT. That pins which calls are PRESENT, and
// where they sit relative to each other, but not how they are COMBINED -- and in this effect
// the combinator is the invariant. Changing the inner `Promise.all` to `Promise.race` is one
// token, it leaves every call present and every index in the same order, and it reinstates
// exactly the race the gate was added to close: the settings GET then fires as soon as EITHER
// prerequisite settles, so on a first send it can still overtake the row write, find no row,
// and release the chat's held edits into the installation defaults.
//
// Measured on this file's parent commit: that edit passes the whole suite, 4045 of 4045.
//
// So these assertions walk the syntax tree instead. Kept out of the text-based file on
// purpose, like tsx-ast.ts and module-stubs.ts: only the few tests that need the TypeScript
// compiler should pay for loading it.

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
  // Neither wait is bounded on its own: the settings chain is a PATCH, and
  // awaitStoredChatThreadWrites settles the row write through settleCurrent, which is
  // Promise.allSettled over work that opens with an unbounded getStoredChatThread. Outside
  // the deadline their time is uncounted, THREAD_PAIRING_WAIT_MS stops bounding the chain it
  // was sized against, and a stalled write becomes "the message was not sent".
  //
  // The structural form of the sibling file's index check: the deadline must CONTAIN the
  // prerequisites, not merely appear before them in the text.
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
