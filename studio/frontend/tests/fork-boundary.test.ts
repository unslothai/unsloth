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

const { useForkBoundaryStore, setForkBoundary, forkBoundaryAnchor } =
  await import("../src/features/chat/stores/fork-boundary-store.ts");

const reset = () =>
  useForkBoundaryStore.setState(
    { boundaryByThreadId: {}, anchorByThreadId: {} },
    false,
  );

const ids = (threadId: string) => [
  ...(useForkBoundaryStore.getState().boundaryByThreadId[threadId]?.messageIds ??
    []),
];

test("a thread with no boundary published has none", () => {
  reset();
  assert.equal(
    useForkBoundaryStore.getState().boundaryByThreadId["missing"],
    undefined,
  );
});

test("a fork's inherited messages and source are kept per thread", () => {
  reset();
  setForkBoundary("fork-a", ["m1", "m2"], "src-a");
  setForkBoundary("fork-b", ["m9"], "src-b");
  const { boundaryByThreadId } = useForkBoundaryStore.getState();
  assert.deepEqual(ids("fork-a"), ["m1", "m2"]);
  assert.equal(boundaryByThreadId["fork-a"].sourceThreadId, "src-a");
  assert.deepEqual(ids("fork-b"), ["m9"]);
  assert.equal(boundaryByThreadId["fork-b"].sourceThreadId, "src-b");
});

test("an omitted source leaves the divider without a link", () => {
  reset();
  setForkBoundary("t", ["msg-1"]);
  assert.equal(
    useForkBoundaryStore.getState().boundaryByThreadId["t"].sourceThreadId,
    null,
  );
});

test("an empty set clears the entry, so a chat with nothing inherited shows no divider", () => {
  reset();
  setForkBoundary("t", ["msg-1"]);
  setForkBoundary("t", []);
  assert.equal(useForkBoundaryStore.getState().boundaryByThreadId["t"], undefined);
});

test("null and undefined clear the entry, so a plain chat shows no divider", () => {
  reset();
  setForkBoundary("t", ["msg-1"]);
  setForkBoundary("t", null);
  assert.equal(useForkBoundaryStore.getState().boundaryByThreadId["t"], undefined);
  setForkBoundary("t", ["msg-1"]);
  setForkBoundary("t", undefined);
  assert.equal(useForkBoundaryStore.getState().boundaryByThreadId["t"], undefined);
});

test("republishing the same boundary keeps the identity, so rows do not re-render", () => {
  reset();
  setForkBoundary("t", ["msg-1", "msg-2"], "src");
  const first = useForkBoundaryStore.getState().boundaryByThreadId;
  // A fresh Set with the same members is the same boundary, so identity has to survive it.
  setForkBoundary("t", ["msg-1", "msg-2"], "src");
  assert.equal(useForkBoundaryStore.getState().boundaryByThreadId, first);
  // A message gone from the set is a real change, and so is a deleted source.
  setForkBoundary("t", ["msg-1"], "src");
  assert.notEqual(useForkBoundaryStore.getState().boundaryByThreadId, first);
  setForkBoundary("t", ["msg-1", "msg-2"], "src");
  const second = useForkBoundaryStore.getState().boundaryByThreadId;
  setForkBoundary("t", ["msg-1", "msg-2"], null);
  assert.notEqual(useForkBoundaryStore.getState().boundaryByThreadId, second);
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

test("the divider is gated on the anchor the tracker resolved", () => {
  const text = findDeclaration("ForkContinuationRule").getText();
  assert.match(text, /useForkBoundaryStore/);
  // Nothing at all when this is not the anchor row.
  assert.match(
    text,
    /if \(anchor === undefined \|\| anchor !== messageId\) return null;/,
  );
  assert.match(text, /anchorByThreadId\[threadId\]/);
  assert.match(text, /Continued from chat/);
});

test("only the tracker selects the message array, never a row", () => {
  // Selecting it per row is what the delete render budget forbids: one delete would
  // re-render every message in the thread along with its action bar.
  const rule = findDeclaration("ForkContinuationRule").getText();
  assert.doesNotMatch(rule, /thread\.messages/);
  const tracker = findDeclaration("useTrackForkBoundaryAnchor").getText();
  assert.match(tracker, /useAuiState\(\(\{ thread \}\) =>/);
  assert.match(tracker, /forkBoundaryAnchor\(thread\.messages, inherited\)/);
  assert.match(tracker, /setForkBoundaryAnchor\(threadId, anchor\)/);
  // Mounted once for the thread, not inside the row slot.
  assert.match(
    threadSource,
    /useTrackForkBoundaryAnchor\(threadId\);/,
  );
});

test("a thread with no active id never claims a boundary", () => {
  const text = findDeclaration("ForkContinuationRule").getText();
  assert.match(text, /threadId === null \? undefined/);
});

test("the label links back to the source chat, and only while there is one", () => {
  const text = findDeclaration("ForkContinuationRule").getText();
  // Same navigation the fork action itself uses.
  assert.match(text, /to: "\/chat",/);
  assert.match(text, /\{ thread: sourceThreadId \}/);
  // A paired source reopens as the comparison it was forked from, not as one of its panes.
  assert.match(text, /source\?\.pairId\s*\?\s*\{ compare: source\.pairId \}/);
  // The local tombstone set is this tab's own, so existence is settled on the way out, and
  // against the backend: getStoredChatThread answers with this browser's legacy row instead.
  assert.match(text, /await readBackendChatThread\(sourceThreadId\)/);
  assert.doesNotMatch(text, /getStoredChatThread/);
  // Only a definite null stops the trip; undefined means the backend could not say, and
  // falls through to the single-chat view rather than claiming a deletion.
  assert.match(text, /source === null\) \{[^}]*toast\.info/s);
  assert.match(text, /sourceThreadId \? \(/);
  // A deleted source keeps the words but drops the button.
  assert.match(text, /<span className=\{labelClass\}>\{label\}<\/span>/);
  // A button, not an anchor: it is in-app navigation, not a URL.
  assert.match(text, /<button\n?\s*type="button"/);
});

// --- which inherited message closes the history, per branch ---------------------

const row = (id: string, role: "user" | "assistant" | "system" = "user") => ({
  id,
  role,
});

test("the divider follows the last inherited message on the branch", () => {
  const inherited = new Set(["m1", "m2", "m3", "m4"]);
  assert.equal(
    forkBoundaryAnchor([row("m1"), row("m2", "assistant"), row("m3"), row("m4", "assistant")], inherited),
    "m4",
  );
  // New turns after the fork do not move it.
  assert.equal(
    forkBoundaryAnchor(
      [row("m1"), row("m2", "assistant"), row("m3"), row("m4", "assistant"), row("n1"), row("n2", "assistant")],
      inherited,
    ),
    "m4",
  );
});

test("editing an inherited message moves the divider up, it does not lose it", () => {
  // assistant-ui keeps the originals and selects a sibling branch, so m3 and m4 are off
  // screen while m1 and m2 are still inherited and visible.
  const inherited = new Set(["m1", "m2", "m3", "m4"]);
  assert.equal(
    forkBoundaryAnchor(
      [row("m1"), row("m2", "assistant"), row("m3-edited"), row("a-new", "assistant")],
      inherited,
    ),
    "m2",
  );
  // Switching back to the original branch restores it, which one stored id cannot do.
  assert.equal(
    forkBoundaryAnchor([row("m1"), row("m2", "assistant"), row("m3"), row("m4", "assistant")], inherited),
    "m4",
  );
});

test("editing the first inherited message leaves no divider", () => {
  // Nothing inherited survives on this branch, so there is no history to close.
  assert.equal(
    forkBoundaryAnchor([row("m1-edited"), row("a-new", "assistant")], new Set(["m1", "m2"])),
    undefined,
  );
});

test("the divider never lands on a message that paints no row", () => {
  // A system message renders as nothing, so it cannot carry the rule.
  assert.equal(
    forkBoundaryAnchor([row("m1"), row("m2", "assistant"), row("m3", "system")], new Set(["m1", "m2", "m3"])),
    "m2",
  );
});

test("a thread with nothing inherited has no anchor", () => {
  assert.equal(forkBoundaryAnchor([row("m1")], undefined), undefined);
  assert.equal(forkBoundaryAnchor([row("m1")], new Set()), undefined);
});

test("the walk stops at the divergence rather than scanning the thread", () => {
  const inherited = new Set(["m1"]);
  let seen = 0;
  const messages = Array.from({ length: 500 }, (_, i) => ({
    get id() {
      seen += 1;
      return i === 0 ? "m1" : `n${i}`;
    },
    role: "user" as const,
  }));

  assert.equal(forkBoundaryAnchor(messages, inherited), "m1");
  // The inherited chain is a prefix, so a fork whose history has grown long does not pay
  // for the whole array on every store notification.
  assert.ok(seen <= 4, `walked ${seen} ids`);
});
