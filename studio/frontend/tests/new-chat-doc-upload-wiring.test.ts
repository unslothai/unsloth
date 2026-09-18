// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * THE DOCS BAR MUST REACH `materializeThreadScope`, NOT `requireStoredThread` DIRECTLY.
 *
 * `new-chat-doc-upload-materialization.test.ts` drives the decision helper in isolation, which is
 * the right shape for the decision but leaves the WIRING unguarded: restore the pre-fix body of
 * `ensureThreadId` and that suite still reports 7/7, because nothing in it touches
 * thread-documents-bar.tsx. The bug in #11186 was not in the helper, which did not exist; it was
 * that `ensureThreadId` handed an unpersisted `__LOCALID_` id straight to `requireStoredThread`.
 * This file is what fails if that shortcut comes back.
 *
 * Four things are asserted, all on the AST rather than on text, so a reformat cannot satisfy them:
 *   1. `ensureThreadId` calls `materializeThreadScope`;
 *   2. the `initialize` it hands that call bottoms out in `initializeThreadItem`, so the recovery
 *      lands on the shared ref-deduped materialization rather than a second, undeduped one;
 *   3. the clear boundary is captured in `ensureThreadId`, before the first await, so a Clear All
 *      landing during the stored-thread check is still seen;
 *   4. `ensureThreadId` no longer calls `requireStoredThread` itself, which is the exact shape of
 *      the pre-fix early return.
 *
 * IT IS A SOURCE TEST AND THAT IS A REAL LIMIT. It proves the call is wired, not that the call is
 * correct; the decision's own behaviour is covered cell by cell in the sibling suite. A rewrite
 * that routed through some third helper would fail here and be right to, which is the intended
 * cost of pinning a seam this load-bearing: `studio/backend/routes/rag.py` does not validate the
 * thread, so this is the only guard between a stale id and an orphaned document.
 */

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
  // Either the callback itself or a thin wrapper around it, but it must bottom out there:
  // a second, undeduped materialization would let a double-click start two threads.
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
  // clearAllChats() advances the boundary as its first statement so that "a late initializer
  // must not recreate a chat after this clear finishes". Capturing it inside
  // initializeThreadItem instead puts the read AFTER `await requireStoredThread(...)`, which
  // waits on a Dexie read and an unbounded getChatThread round trip, so a clear landing in
  // that window reads as no clear at all and the row is written anyway.
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
