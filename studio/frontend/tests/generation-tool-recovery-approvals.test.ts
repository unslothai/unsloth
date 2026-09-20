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

test("a call that parks after the tab reattaches arms its own card", () => {
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

test("a call that parked BEFORE the tab closed is re-raised from the seed", () => {
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
  recovery.armSeededApprovals("sess-1");
  assert.deepEqual(spy.registered, [
    { partId: "sess-1:thread-1:appr-7", approvalId: "appr-7", sessionId: "sess-1" },
  ]);
});

test("a finished card is never re-armed from the seed", () => {
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
  createGenerationToolRecovery(carried, "run-1", 40, spy.hooks).armSeededApprovals("sess-1");
  assert.deepEqual(spy.registered, [], "an answered call has a result; it is not waiting on anyone");
});

test("the seed and the frame can name the same call without raising two cards", () => {
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
  recovery.armSeededApprovals("sess-1");
  recovery.apply(parkedStart("appr-1"), 5, 9, "sess-1");

  assert.equal(spy.live.size, 1, "one card, however many times it was named");
  assert.equal(carried.length, 1, "and the frame folded into the saved card, not a second one");
  assert.equal(spy.live.get("sess-1:thread-1:appr-1")?.approvalId, "appr-1");
});

test("the decision goes away when the call gets a result", () => {
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

test("a call that never needed a decision arms nothing", () => {
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

test("recovery still works for a caller that passes no confirmation hooks", () => {
  // The hooks are optional: every other caller of this module keeps its old shape.
  const carried: { at: number; part: unknown }[] = [];
  const recovery = createGenerationToolRecovery(carried, "run-1", 0);
  recovery.apply(parkedStart("appr-1"), 0, 1, "sess-1");
  recovery.armSeededApprovals("sess-1");
  assert.equal(carried.length, 1);
  assert.equal(
    (carried[0].part as Record<string, unknown>).toolApprovalId,
    "appr-1",
    "the approval id still lands on the part",
  );
});
