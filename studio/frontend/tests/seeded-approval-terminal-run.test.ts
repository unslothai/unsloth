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

// ── Arming must not hold the event stream closed ────────────────────────────
// followChatGenerationRun yields its snapshot BEFORE it opens /events
// (chat-generation-api.ts: `yield { run, source: "snapshot" }` precedes the
// streamChatGenerationEvents loop), so the consumer is what decides when that stream opens.
// /events is also the only thing that marks the run attended server-side
// (state/run_subscribers.py), and attendance is what stops the park ceiling denying an
// approval. Awaiting the seeded-approval check at the call site therefore serialises the
// "is it still pending?" request AHEAD of the stream that says someone is watching, so a tab
// returning near the ceiling can have its approval expire during that very request and be
// handed buttons that can only 404.
//
// Pinned on the call site for the same reason as the guard above: the follow loop needs a live
// EventSource, so no behavioural test in this suite reaches it.

/** Walk up from a node looking for an `await` that directly wraps it. */
const isDirectlyAwaited = (node: ts.Node): boolean => {
  let current: ts.Node | undefined = node.parent;
  while (current) {
    if (ts.isAwaitExpression(current)) return true;
    // Stop at the first construct that is not a transparent wrapper: anything further up
    // awaits a different expression, not this call.
    if (
      ts.isExpressionStatement(current) ||
      ts.isVariableDeclaration(current) ||
      ts.isBinaryExpression(current) ||
      ts.isBlock(current)
    ) {
      return false;
    }
    current = current.parent;
  }
  return false;
};

test("the seeded-approval check does not block the event stream from opening", () => {
  for (const call of seededApprovalCalls()) {
    assert.ok(
      !isDirectlyAwaited(call),
      "armSeededApprovals must NOT be awaited at its call site. The follower is suspended " +
        "at followChatGenerationRun's snapshot yield, so awaiting here delays " +
        "streamChatGenerationEvents, which is the only thing that marks the run attended " +
        "(state/run_subscribers.py). A tab returning near UNSLOTH_STUDIO_TOOL_APPROVAL_TIMEOUT_S " +
        "would then lose the approval during the request that checks whether it is still pending. " +
        "Assign the promise and join it before disarmAll instead.",
    );
  }
});


// ── The disarm has to cover every exit, not just the terminal one ───────────
// A permanent follower error -- another tab deletes the thread, the run row cascades, the next
// request 404s -- throws straight past the terminal branch and is swallowed by the outer catch. A
// card armed from the seed would then sit in the global toolConfirmations store for the rest of the
// session. Nothing renders buttons over a finished part, but `soleRequest` counts entries, so a
// later real approval reads as non-sole and silently loses its Enter and Escape chords.

test("disarmAll runs in the recovery's finally, not only on the terminal path", () => {
  const text = readFileSync(SOURCE, "utf8");
  const disarms = [...text.matchAll(/toolRecovery\.disarmAll\(\)/g)];
  assert.equal(
    disarms.length,
    1,
    "expected exactly one disarmAll call site; more than one means the terminal-path copy came " +
      "back and the two can disagree about ordering against the seed join",
  );
  // The call must sit inside a `finally {`, which is the only block every exit reaches.
  const before = text.slice(0, disarms[0]!.index!);
  const lastFinally = before.lastIndexOf("} finally {");
  const lastTerminalIf = before.lastIndexOf("isTerminalChatGenerationRun(update.run)");
  assert.ok(
    lastFinally > lastTerminalIf,
    "disarmAll must sit in the recovery's finally block. Under the terminal `if` it is skipped " +
      "whenever the follower throws, which leaks the armed card into the global store.",
  );
});

test("the seed is joined before the disarm, wherever the disarm lives", () => {
  const text = readFileSync(SOURCE, "utf8");
  const disarmIndex = text.indexOf("toolRecovery.disarmAll()");
  assert.ok(disarmIndex > 0, "disarmAll call site not found");
  assert.match(
    text.slice(0, disarmIndex),
    /await seededApprovals/,
    "the seeded-approval promise must be joined before disarmAll, or an arm that resolves " +
      "late re-raises buttons on a run that is already over",
  );
});
