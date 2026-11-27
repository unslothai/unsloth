// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A call approved in another tab or device: the backend sends tool_approved, and every follower drops
// its Allow / Deny (and the sidebar's waiting dot) then, instead of at tool_end after the tool has run.

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "@typescript/typescript6";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { normalizeChatGenerationChunkPayload } = await import(
  "../src/features/chat/api/chat-generation-api.ts"
);

const frame = {
  type: "tool_approved",
  tool_name: "python",
  tool_call_id: "call_0",
  approval_id: "approval-1",
};

test("a persisted tool_approved frame is re-tagged like the live stream", () => {
  assert.deepEqual(normalizeChatGenerationChunkPayload(frame), {
    _toolEvent: frame,
  });
});

test("the live stream routes tool_approved off the transcript", () => {
  const src = readSrc("features/chat/api/chat-api.ts");
  assert.match(
    src,
    /parsed\.type === "tool_approved"\)\s*\) \{\s*yield \{ _toolEvent: parsed \}/,
  );
});

// The adapter's handler runs as written: its block is cut out of chat-adapter.ts and run against the
// same maps the tool_start branch fills.
function liveHandler() {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const startAt = adapter.indexOf('if (toolEvent.type === "tool_approved") {');
  const endAt = adapter.indexOf(
    "// Persist container_id onto the thread",
    startAt,
  );
  assert.ok(startAt >= 0 && endAt > startAt);
  const source = `function receive(events) { for (const toolEvent of events) { ${adapter.slice(startAt, endAt)} } }`;
  return ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
}

function run(
  confirmationIds: [string, string][],
  parts: Record<string, unknown>[],
  events: Record<string, unknown>[],
) {
  const cleared: string[] = [];
  const context = vm.createContext({
    toolConfirmationIdsByBackendId: new Map(confirmationIds),
    toolCallParts: parts,
    useChatRuntimeStore: {
      getState: () => ({
        clearToolConfirmation: (id: string) => cleared.push(id),
      }),
    },
  });
  vm.runInContext(liveHandler(), context);
  context.receive(events);
  return cleared;
}

test("the live stream clears the parked card the approval names", () => {
  assert.deepEqual(
    run(
      [["call_0", "session:thread:approval-1"]],
      [{ toolCallId: "session:thread:approval-1", toolApprovalId: "approval-1" }],
      [frame],
    ),
    ["session:thread:approval-1"],
  );
});

test("an id-less call is found by its approval id", () => {
  assert.deepEqual(
    run(
      [],
      [
        { toolCallId: "other", toolApprovalId: "approval-0" },
        { toolCallId: "session:thread:approval-1", toolApprovalId: "approval-1" },
      ],
      [{ ...frame, tool_call_id: "" }],
    ),
    ["session:thread:approval-1"],
  );
});

test("an approval for a call this stream never parked clears nothing", () => {
  assert.deepEqual(
    run([], [{ toolCallId: "call_9:live" }], [{ ...frame, tool_call_id: "call_9", approval_id: "" }]),
    [],
  );
});
