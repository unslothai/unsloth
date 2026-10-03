// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* Guards the WIRING, which new-chat-doc-upload-materialization.test.ts does not: restore the
 * pre-fix body of `ensureThreadId` and that suite still reports 7/7, because it drives the helper
 * in isolation and never reads thread-documents-bar.tsx. Source test, so it proves the seam is
 * wired, not that the decision is right. */

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import { readText } from "./helpers/kit.ts";

const REL = "../src/features/rag/components/thread-documents-bar.tsx";

const source = ts.createSourceFile(
  REL,
  readText(REL),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

/** The `const <name> = useCallback(...)` initializer, so the search stays inside one function. */
function namedCallback(name: string): ts.Node {
  let found: ts.Node | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === name &&
      node.initializer
    ) {
      found = node.initializer;
      return;
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  assert.ok(found, `${name} not found in ${REL}`);
  return found;
}

function callsTo(scope: ts.Node, callee: string): ts.CallExpression[] {
  const out: ts.CallExpression[] = [];
  const visit = (node: ts.Node): void => {
    if (
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === callee
    ) {
      out.push(node);
    }
    ts.forEachChild(node, visit);
  };
  visit(scope);
  return out;
}

test("ensureThreadId routes through materializeThreadScope", () => {
  const calls = callsTo(
    namedCallback("ensureThreadId"),
    "materializeThreadScope",
  );
  assert.equal(
    calls.length,
    1,
    "ensureThreadId must materialize through the decision helper exactly once",
  );
});

test("the materialization it passes is the ref-deduped initializeThreadItem", () => {
  const [call] = callsTo(
    namedCallback("ensureThreadId"),
    "materializeThreadScope",
  );
  const [arg] = call.arguments;
  assert.ok(
    arg && ts.isObjectLiteralExpression(arg),
    "materializeThreadScope takes one object literal",
  );
  const initialize = arg.properties.find(
    (p): p is ts.PropertyAssignment =>
      ts.isPropertyAssignment(p) &&
      ts.isIdentifier(p.name) &&
      p.name.text === "initialize",
  );
  assert.ok(initialize, "materializeThreadScope must be given an `initialize`");
  // Identifier or thin wrapper, but it must bottom out there: a second, undeduped
  // materialization would let a double-click start two threads.
  const reachesIt =
    (ts.isIdentifier(initialize.initializer) &&
      initialize.initializer.text === "initializeThreadItem") ||
    callsTo(initialize.initializer, "initializeThreadItem").length === 1;
  assert.ok(
    reachesIt,
    "`initialize` must reach the shared ref-deduped initializeThreadItem",
  );
});

test("the clear boundary is captured in ensureThreadId, before anything can yield", () => {
  // Captured inside initializeThreadItem instead, the read lands AFTER an unbounded
  // getChatThread round trip, so a Clear All arriving in that window reads as no clear and the
  // row is written anyway (clear-all-chats.ts advances the boundary as its first statement).
  const scope = namedCallback("ensureThreadId");
  const captures = (node: ts.Node): number => {
    let n = 0;
    const visit = (x: ts.Node): void => {
      if (
        ts.isCallExpression(x) &&
        ts.isPropertyAccessExpression(x.expression) &&
        ts.isIdentifier(x.expression.expression) &&
        x.expression.expression.text === "chatHistoryClearBoundary" &&
        x.expression.name.text === "capture"
      ) {
        n += 1;
      }
      ts.forEachChild(x, visit);
    };
    visit(node);
    return n;
  };
  assert.equal(
    captures(scope),
    1,
    "ensureThreadId must capture chatHistoryClearBoundary itself and hand it down",
  );
});

test("ensureThreadId no longer calls requireStoredThread itself", () => {
  // The pre-fix body was `requireStoredThread(effectiveThreadId).then(() => effectiveThreadId)`,
  // which is what threw `Thread __LOCALID_xxx was not persisted` on a brand-new chat (#11186).
  const direct = callsTo(
    namedCallback("ensureThreadId"),
    "requireStoredThread",
  );
  assert.equal(
    direct.length,
    0,
    "requireStoredThread must be reached through materializeThreadScope, which can recover " +
      "an unpersisted __LOCALID_ id, not called directly",
  );
});
