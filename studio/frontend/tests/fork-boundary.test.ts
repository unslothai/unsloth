// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The store behind the "Continued from chat" divider, plus the JSX wiring that reads it.
// The store is plain .ts, so those claims run the code. thread.tsx is .tsx and this runner
// cannot execute JSX, so its claims go through the TypeScript AST rather than a substring
// of the source: a substring passes on broken code and fails on a reformat.

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { useForkBoundaryStore, setForkBoundary } = await import(
  "../src/features/chat/stores/fork-boundary-store.ts"
);

const reset = () =>
  useForkBoundaryStore.setState({ boundaryByThreadId: {} }, false);

test("a thread with no boundary published has none", () => {
  reset();
  assert.equal(
    useForkBoundaryStore.getState().boundaryByThreadId["missing"],
    undefined,
  );
});

test("a fork's boundary and source are kept per thread", () => {
  reset();
  setForkBoundary("fork-a", "msg-1", "src-a");
  setForkBoundary("fork-b", "msg-2", "src-b");
  const { boundaryByThreadId } = useForkBoundaryStore.getState();
  assert.deepEqual(boundaryByThreadId["fork-a"], {
    messageId: "msg-1",
    sourceThreadId: "src-a",
  });
  assert.deepEqual(boundaryByThreadId["fork-b"], {
    messageId: "msg-2",
    sourceThreadId: "src-b",
  });
});

test("an omitted source leaves the divider without a link", () => {
  reset();
  setForkBoundary("t", "msg-1");
  assert.equal(
    useForkBoundaryStore.getState().boundaryByThreadId["t"].sourceThreadId,
    null,
  );
});

test("null and undefined clear the entry, so a plain chat shows no divider", () => {
  reset();
  setForkBoundary("t", "msg-1");
  setForkBoundary("t", null);
  assert.equal(useForkBoundaryStore.getState().boundaryByThreadId["t"], undefined);
  setForkBoundary("t", "msg-1");
  setForkBoundary("t", undefined);
  assert.equal(useForkBoundaryStore.getState().boundaryByThreadId["t"], undefined);
});

test("republishing the same boundary keeps the identity, so rows do not re-render", () => {
  reset();
  setForkBoundary("t", "msg-1", "src");
  const first = useForkBoundaryStore.getState().boundaryByThreadId;
  setForkBoundary("t", "msg-1", "src");
  assert.equal(useForkBoundaryStore.getState().boundaryByThreadId, first);
  // A source that has since been deleted is a real change.
  setForkBoundary("t", "msg-1", null);
  assert.notEqual(useForkBoundaryStore.getState().boundaryByThreadId, first);
});

test("clearing a thread that has no entry keeps the identity", () => {
  reset();
  const first = useForkBoundaryStore.getState().boundaryByThreadId;
  setForkBoundary("never-set", null);
  assert.equal(useForkBoundaryStore.getState().boundaryByThreadId, first);
});

// --- the wiring in thread.tsx -------------------------------------------------

const threadSource = readSrc("components/assistant-ui/thread.tsx");
const sourceFile = ts.createSourceFile(
  "thread.tsx",
  threadSource,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);

function findDeclaration(name: string): ts.VariableDeclaration {
  let found: ts.VariableDeclaration | undefined;
  const walk = (node: ts.Node) => {
    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === name
    ) {
      found = node;
    }
    ts.forEachChild(node, walk);
  };
  ts.forEachChild(sourceFile, walk);
  assert.ok(found, `${name} should be declared in thread.tsx`);
  return found;
}

test("ThreadMessage renders the divider after every message kind", () => {
  const text = findDeclaration("ThreadMessage").getText();
  assert.match(text, /<ForkContinuationRule \/>/);
  // After the body, so the rule closes the inherited history rather than heading it.
  const body = text.indexOf("{body}");
  assert.ok(body >= 0, "ThreadMessage should render the message body");
  assert.ok(
    body < text.indexOf("<ForkContinuationRule />"),
    "the rule should follow the message body",
  );
});

test("the divider is gated on the active thread's boundary", () => {
  const text = findDeclaration("ForkContinuationRule").getText();
  assert.match(text, /useForkBoundaryStore/);
  assert.match(text, /boundary\.messageId !== messageId/);
  // Nothing at all when this is not the boundary row.
  assert.match(text, /if \(!boundary \|\| boundary\.messageId !== messageId\) return null;/);
  assert.match(text, /Continued from chat/);
});

test("a thread with no active id never claims a boundary", () => {
  const text = findDeclaration("ForkContinuationRule").getText();
  assert.match(text, /threadId === null \? undefined/);
});

test("the label links back to the source chat, and only while there is one", () => {
  const text = findDeclaration("ForkContinuationRule").getText();
  // Same navigation the fork action itself uses.
  assert.match(text, /to: "\/chat",\s*search: \{ thread: sourceThreadId \}/);
  // The local tombstone set is this tab's own, so existence is settled on the way out, and
  // against the backend: getStoredChatThread answers with this browser's legacy row instead.
  assert.match(text, /await chatThreadExistsOnBackend\(sourceThreadId\)/);
  assert.doesNotMatch(text, /getStoredChatThread/);
  // Only a definite "no" stops the trip; undefined means the backend could not say.
  assert.match(text, /=== false\) \{[^}]*toast\.info/s);
  assert.match(text, /sourceThreadId \? \(/);
  // A deleted source keeps the words but drops the button.
  assert.match(text, /<span className=\{labelClass\}>\{label\}<\/span>/);
  // A button, not an anchor: it is in-app navigation, not a URL.
  assert.match(text, /<button\n?\s*type="button"/);
});
