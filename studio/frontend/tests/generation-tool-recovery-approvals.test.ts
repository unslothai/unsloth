// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { createGenerationToolRecovery } =
  await import("../src/features/chat/utils/generation-tool-recovery.ts");

type Registered = { partId: string; approvalId: string; sessionId: string };

function spyConfirmations() {
  const registered: Registered[] = [];
  const resolved: string[] = [];
  const live = new Map<string, Registered>();
  return {
    registered,
    resolved,
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
  const recovery = createGenerationToolRecovery(
    carried,
    "run-1",
    40,
    spy.hooks,
  );

  assert.deepEqual(
    spy.registered,
    [],
    "nothing is armed before the session is known",
  );
  await recovery.armSeededApprovals("sess-1");
  assert.deepEqual(spy.registered, [
    {
      partId: "sess-1:thread-1:appr-7",
      approvalId: "appr-7",
      sessionId: "sess-1",
    },
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
  await createGenerationToolRecovery(
    carried,
    "run-1",
    40,
    spy.hooks,
  ).armSeededApprovals("sess-1");
  assert.deepEqual(
    spy.registered,
    [],
    "an answered call has a result; it is not waiting on anyone",
  );
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
  assert.equal(
    carried.length,
    1,
    "and the frame folded into the saved card, not a second one",
  );
  assert.equal(spy.live.get("sess-1:thread-1:appr-1")?.approvalId, "appr-1");
});

test("the decision goes away when the call gets a result", async () => {
  const spy = spyConfirmations();
  const carried: { at: number; part: unknown }[] = [];
  const recovery = createGenerationToolRecovery(carried, "run-1", 0, spy.hooks);

  recovery.apply(parkedStart("appr-1"), 0, 1, "sess-1");
  const partId = (carried[0].part as Record<string, unknown>)
    .toolCallId as string;
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
  const spy = spyConfirmations();
  const recovery = createGenerationToolRecovery(
    seededParkedCard(),
    "run-1",
    40,
    spy.hooks,
  );

  await recovery.armSeededApprovals("sess-1");
  assert.equal(spy.registered.length, 1);
  assert.equal(spy.live.size, 1, "the card is live before the run ends");

  recovery.disarmAll();
  assert.deepEqual(spy.resolved, ["sess-1:thread-1:appr-9"]);
  assert.equal(spy.live.size, 0, "no card outlives its run");

  recovery.disarmAll();
  assert.deepEqual(spy.resolved, ["sess-1:thread-1:appr-9"]);
});

test("the run-level sweep only takes down cards this recovery armed", async () => {
  const spy = spyConfirmations();
  const recovery = createGenerationToolRecovery(
    seededParkedCard(),
    "run-1",
    40,
    spy.hooks,
  );

  recovery.disarmAll();
  assert.deepEqual(spy.resolved, []);
});

test("a call the user already answered is not re-armed on reopen", async () => {
  const spy = spyConfirmations();
  const recovery = createGenerationToolRecovery(
    seededParkedCard(),
    "run-1",
    40,
    spy.hooks,
  );

  await recovery.armSeededApprovals("sess-1", async () => false);
  assert.deepEqual(
    spy.registered,
    [],
    "an answered call must not get its buttons back",
  );
  assert.equal(spy.live.size, 0);
});

test("a call that really is still parked is armed as before", async () => {
  const spy = spyConfirmations();
  const recovery = createGenerationToolRecovery(
    seededParkedCard(),
    "run-1",
    40,
    spy.hooks,
  );

  await recovery.armSeededApprovals("sess-1", async () => true);
  assert.equal(spy.registered.length, 1);
  assert.equal(spy.live.size, 1);
});

test("a check that cannot be answered still arms the card", async () => {
  const spy = spyConfirmations();
  const recovery = createGenerationToolRecovery(
    seededParkedCard(),
    "run-1",
    40,
    spy.hooks,
  );

  await recovery.armSeededApprovals("sess-1", async () => {
    throw new Error("network down");
  });
  assert.equal(
    spy.registered.length,
    1,
    "a failed check must not cost a parked call its buttons",
  );
});

// The status check and the replay run concurrently, so `tool_end` can fold a card mid-request;
// arming afterwards must not register an already-finished call.

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
  const recovery = createGenerationToolRecovery(
    carried,
    "run-1",
    40,
    spy.hooks,
  );

  let release: (value: boolean) => void = () => {};
  const gate = new Promise<boolean>((resolve) => {
    release = resolve;
  });
  const arming = recovery.armSeededApprovals("sess-1", () => gate);

  recovery.apply(finishedEnd("appr-9"), 0, 41, "sess-1");
  release(true);
  await arming;

  assert.deepEqual(
    spy.registered,
    [],
    "the card gained a result while the status request was in flight, so arming it leaves a " +
      "store entry nothing will ever clear until the run ends",
  );
});

test("a card still unresolved when the status request returns is armed as before", async () => {
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
  const recovery = createGenerationToolRecovery(
    carried,
    "run-1",
    40,
    spy.hooks,
  );
  await recovery.armSeededApprovals("sess-1", async () => true);
  assert.deepEqual(spy.registered, [
    {
      partId: "sess-1:thread-1:appr-10",
      approvalId: "appr-10",
      sessionId: "sess-1",
    },
  ]);
});

test("a reopened tab rebuilds an MCP App result, so the widget mounts again", () => {
  const carried: { at: number; part: unknown }[] = [];
  const replay = createGenerationToolRecovery(carried, "run-1", 0).apply;
  const ui = {
    resourceUri: "ui://weather/view.html",
    structuredContent: { c: 21 },
  };
  const image = { data: "AAAA", mimeType: "image/png" };
  replay(
    {
      type: "tool_start",
      tool_call_id: "c1",
      tool_name: "mcp__s1__get_weather",
      arguments: {},
    },
    0,
    1,
  );
  replay(
    {
      type: "tool_end",
      tool_call_id: "c1",
      result: `21 C\n__MCP_UI__:${JSON.stringify(ui)}\n__MCP_IMAGES__:${JSON.stringify([image])}`,
    },
    1,
    2,
  );
  assert.deepEqual((carried[0].part as { result: unknown }).result, {
    text: "21 C",
    images: [image],
    ui,
  });
});
