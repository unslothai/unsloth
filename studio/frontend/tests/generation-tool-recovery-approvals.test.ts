// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { createGenerationToolRecovery } = await import(
  "../src/features/chat/utils/generation-tool-recovery.ts"
);

// A durable tool turn parks the backend on a human decision and keeps waiting for the returning
// session (state/tool_approvals.wait_tool_decision). The reopened tab IS that session, so the card it
// rebuilds has to offer Approve/Deny -- ToolConfirmationControls renders them only for a card the
// confirmation store knows about. These pin both ways a parked call can reach a reopened tab: as a
// frame above the cursor, and as a seed whose frame will never be re-folded.

type Registered = { partId: string; approvalId: string; sessionId: string };

function spyConfirmations() {
  const registered: Registered[] = [];
  const resolved: string[] = [];
  const live = new Map<string, Registered>();
  return {
    registered,
    resolved,
    /** What the store actually holds, after registers and resolves have both been applied. */
    live,
    hooks: {
      register: (partId: string, approvalId: string, sessionId: string) => {
        registered.push({ partId, approvalId, sessionId });
        live.set(partId, { partId, approvalId, sessionId });
      },
      resolve: (partId: string) => {
        resolved.push(partId);
        live.delete(partId);
      },
    },
  };
}

const parkedStart = (approvalId: string) => ({
  _toolEvent: {
    type: "tool_start",
    tool_name: "terminal",
    tool_call_id: "call-1",
    approval_id: approvalId,
    awaiting_confirmation: true,
    arguments: { command: "echo hi" },
  },
});

test("a call that parks after the tab reattaches arms its own card", async () => {
  const spy = spyConfirmations();
  const carried: { at: number; part: unknown }[] = [];
  const recovery = createGenerationToolRecovery(carried, "run-1", 0, spy.hooks);

  recovery.apply(parkedStart("appr-1"), 0, 1, "sess-1");

  const part = carried[0].part as Record<string, unknown>;
  assert.equal(spy.registered.length, 1);
  assert.deepEqual(spy.registered[0], {
    partId: part.toolCallId as string,
    approvalId: "appr-1",
    sessionId: "sess-1",
  });
});

test("a call that parked BEFORE the tab closed is re-raised from the seed", async () => {
  // The saved card is all that is left of it: its tool_start sits at or below the cursor, so no
  // frame re-folds it. This is the case that made a reopened tab show a spinner with no buttons.
  const spy = spyConfirmations();
  const carried = [
    {
      at: 12,
      part: {
        type: "tool-call",
        toolCallId: "sess-1:thread-1:appr-7",
        toolName: "terminal",
        backendToolCallId: "call-7",
        toolApprovalId: "appr-7",
        args: { command: "echo hi" },
      },
    },
  ];
  const recovery = createGenerationToolRecovery(carried, "run-1", 40, spy.hooks);

  assert.deepEqual(spy.registered, [], "nothing is armed before the session is known");
  await recovery.armSeededApprovals("sess-1");
  assert.deepEqual(spy.registered, [
    { partId: "sess-1:thread-1:appr-7", approvalId: "appr-7", sessionId: "sess-1" },
  ]);
});

test("a finished card is never re-armed from the seed", async () => {
  const spy = spyConfirmations();
  const carried = [
    {
      at: 0,
      part: {
        type: "tool-call",
        toolCallId: "sess-1:thread-1:appr-8",
        toolName: "terminal",
        toolApprovalId: "appr-8",
        result: "hi",
      },
    },
  ];
  await createGenerationToolRecovery(carried, "run-1", 40, spy.hooks).armSeededApprovals("sess-1");
  assert.deepEqual(spy.registered, [], "an answered call has a result; it is not waiting on anyone");
});

test("the seed and the frame can name the same call without raising two cards", async () => {
  const spy = spyConfirmations();
  const carried = [
    {
      at: 5,
      part: {
        type: "tool-call",
        toolCallId: "sess-1:thread-1:appr-1",
        toolName: "terminal",
        backendToolCallId: "call-1",
        toolApprovalId: "appr-1",
      },
    },
  ];
  const recovery = createGenerationToolRecovery(carried, "run-1", 0, spy.hooks);
  await recovery.armSeededApprovals("sess-1");
  recovery.apply(parkedStart("appr-1"), 5, 9, "sess-1");

  assert.equal(spy.live.size, 1, "one card, however many times it was named");
  assert.equal(carried.length, 1, "and the frame folded into the saved card, not a second one");
  assert.equal(spy.live.get("sess-1:thread-1:appr-1")?.approvalId, "appr-1");
});

test("the decision goes away when the call gets a result", async () => {
  const spy = spyConfirmations();
  const carried: { at: number; part: unknown }[] = [];
  const recovery = createGenerationToolRecovery(carried, "run-1", 0, spy.hooks);

  recovery.apply(parkedStart("appr-1"), 0, 1, "sess-1");
  const partId = (carried[0].part as Record<string, unknown>).toolCallId as string;
  assert.equal(spy.live.size, 1);

  recovery.apply(
    {
      _toolEvent: {
        type: "tool_end",
        tool_name: "terminal",
        tool_call_id: "call-1",
        result: "hi\n",
      },
    },
    0,
    2,
    "sess-1",
  );
  assert.deepEqual(spy.resolved, [partId]);
  assert.equal(spy.live.size, 0, "no Approve/Deny over a finished result");
});

test("a call that never needed a decision arms nothing", async () => {
  const spy = spyConfirmations();
  const carried: { at: number; part: unknown }[] = [];
  createGenerationToolRecovery(carried, "run-1", 0, spy.hooks).apply(
    {
      _toolEvent: {
        type: "tool_start",
        tool_name: "web_search",
        tool_call_id: "call-2",
        arguments: { q: "x" },
      },
    },
    0,
    1,
    "sess-1",
  );
  assert.deepEqual(spy.registered, []);
});

test("recovery still works for a caller that passes no confirmation hooks", async () => {
  // The hooks are optional: every other caller of this module keeps its old shape.
  const carried: { at: number; part: unknown }[] = [];
  const recovery = createGenerationToolRecovery(carried, "run-1", 0);
  recovery.apply(parkedStart("appr-1"), 0, 1, "sess-1");
  await recovery.armSeededApprovals("sess-1");
  assert.equal(carried.length, 1);
  assert.equal(
    (carried[0].part as Record<string, unknown>).toolApprovalId,
    "appr-1",
    "the approval id still lands on the part",
  );
});

const seededParkedCard = () => [
  {
    at: 12,
    part: {
      type: "tool-call",
      toolCallId: "sess-1:thread-1:appr-9",
      toolName: "terminal",
      backendToolCallId: "call-9",
      toolApprovalId: "appr-9",
      args: { command: "echo hi" },
    },
  },
];

test("a run that ends without a tool_end still takes its cards down", async () => {
  // The backend failed or restarted while the call was parked, so no tool_end is ever emitted and
  // the per-call disarm never runs. Without a run-level sweep the card outlives its own run:
  // buttons on screen over a dead run, and a decision that can only 404 because the pending slot
  // went with the restart. The terminal guard at the call site covers a run that was ALREADY
  // terminal when recovery attached; this covers one that gets there afterwards.
  const spy = spyConfirmations();
  const recovery = createGenerationToolRecovery(seededParkedCard(), "run-1", 40, spy.hooks);

  await recovery.armSeededApprovals("sess-1");
  assert.equal(spy.registered.length, 1);
  assert.equal(spy.live.size, 1, "the card is live before the run ends");

  recovery.disarmAll();
  assert.deepEqual(spy.resolved, ["sess-1:thread-1:appr-9"]);
  assert.equal(spy.live.size, 0, "no card outlives its run");

  // Idempotent: the follow loop can see more than one terminal snapshot, and a second sweep must
  // not reach for a card this recovery no longer owns.
  recovery.disarmAll();
  assert.deepEqual(spy.resolved, ["sess-1:thread-1:appr-9"]);
});

test("the run-level sweep only takes down cards this recovery armed", async () => {
  // Same scoping rule as the per-call disarm: a recovery shares the store with whatever else is on
  // screen, so a card it never raised is not its business to resolve.
  const spy = spyConfirmations();
  const recovery = createGenerationToolRecovery(seededParkedCard(), "run-1", 40, spy.hooks);

  recovery.disarmAll();                       // armed nothing, so it resolves nothing
  assert.deepEqual(spy.resolved, []);
});

test("a call the user already answered is not re-armed on reopen", async () => {
  // The third way a reopened tab ended up with buttons that cannot work. Approve a slow tool, close
  // the tab before tool_end, reopen: the saved card still has no result and still carries its
  // approval id, because the result only arrives with tool_end. Arming it puts Approve/Deny over a
  // call that is already executing, and every press 404s. The card's shape cannot tell the two
  // apart, so the server is asked.
  const spy = spyConfirmations();
  const recovery = createGenerationToolRecovery(seededParkedCard(), "run-1", 40, spy.hooks);

  await recovery.armSeededApprovals("sess-1", async () => false);
  assert.deepEqual(spy.registered, [], "an answered call must not get its buttons back");
  assert.equal(spy.live.size, 0);
});

test("a call that really is still parked is armed as before", async () => {
  // The control for the test above. If the check were wired the wrong way round, the feature would
  // stop working entirely and the negative test alone would still pass.
  const spy = spyConfirmations();
  const recovery = createGenerationToolRecovery(seededParkedCard(), "run-1", 40, spy.hooks);

  await recovery.armSeededApprovals("sess-1", async () => true);
  assert.equal(spy.registered.length, 1);
  assert.equal(spy.live.size, 1);
});

test("a check that cannot be answered still arms the card", async () => {
  // Offline, or a backend too old to have the route. Losing the buttons on a call that really is
  // parked is the worse failure, so an unanswerable question falls back to the old behaviour.
  const spy = spyConfirmations();
  const recovery = createGenerationToolRecovery(seededParkedCard(), "run-1", 40, spy.hooks);

  await recovery.armSeededApprovals("sess-1", async () => {
    throw new Error("network down");
  });
  assert.equal(spy.registered.length, 1, "a failed check must not cost a parked call its buttons");
});

// ── The check and the replay now run concurrently ───────────────────────────
// runtime-provider no longer awaits armSeededApprovals at its call site: awaiting held the
// /events stream closed, and that stream is the only thing marking the run attended server-side,
// so a tab returning near the park ceiling lost its approval during the very request asking
// whether it was still pending. The cost of un-blocking it is this window: `tool_end` can fold a
// card while its status request is in flight. disarmApproval is a no-op then, because nothing is
// armed yet, so arming afterwards registers an already-finished call. Nothing renders buttons over
// it (a part with a result is not `awaiting`), but the store entry survives until disarmAll, which
// is long enough to make a LATER approval non-sole and silently drop its Enter/Escape chord.

const finishedEnd = (approvalId: string) => ({
  _toolEvent: {
    type: "tool_end",
    tool_name: "terminal",
    tool_call_id: "call-9",
    approval_id: approvalId,
    result: "4.0G\t/home/u/models",
  },
});

test("a card finished while its status request was in flight is not armed afterwards", async () => {
  const spy = spyConfirmations();
  const carried = [
    {
      at: 12,
      part: {
        type: "tool-call",
        toolCallId: "sess-1:thread-1:appr-9",
        toolName: "terminal",
        backendToolCallId: "call-9",
        toolApprovalId: "appr-9",
        args: { command: "du -sh ~/models" },
      },
    },
  ];
  const recovery = createGenerationToolRecovery(carried, "run-1", 40, spy.hooks);

  // Held open so the replay lands strictly between the answer and the resume, which is the race.
  let release: (value: boolean) => void = () => {};
  const gate = new Promise<boolean>((resolve) => {
    release = resolve;
  });
  const arming = recovery.armSeededApprovals("sess-1", () => gate);

  // Another tab answered it, or it expired: the run streams tool_end while we are still waiting.
  recovery.apply(finishedEnd("appr-9"), 0, 41, "sess-1");
  release(true); // the backend still said "pending", because it was, when it was asked
  await arming;

  assert.deepEqual(
    spy.registered,
    [],
    "the card gained a result while the status request was in flight, so arming it leaves a " +
      "store entry nothing will ever clear until the run ends",
  );
});

test("a card still unresolved when the status request returns is armed as before", async () => {
  // The guard above must not cost the case the feature exists for.
  const spy = spyConfirmations();
  const carried = [
    {
      at: 12,
      part: {
        type: "tool-call",
        toolCallId: "sess-1:thread-1:appr-10",
        toolName: "terminal",
        backendToolCallId: "call-10",
        toolApprovalId: "appr-10",
        args: { command: "du -sh ~/models" },
      },
    },
  ];
  const recovery = createGenerationToolRecovery(carried, "run-1", 40, spy.hooks);
  await recovery.armSeededApprovals("sess-1", async () => true);
  assert.deepEqual(spy.registered, [
    { partId: "sess-1:thread-1:appr-10", approvalId: "appr-10", sessionId: "sess-1" },
  ]);
});
