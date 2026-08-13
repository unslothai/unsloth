// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { recordedSandboxSessionId } = await import(
  "../src/features/chat/utils/recorded-sandbox-session.ts"
);

type Message = Parameters<typeof recordedSandboxSessionId>[0][number];

function toolMessage(sessionId: string): Message {
  return {
    id: "m2",
    threadId: "thread-1",
    role: "assistant",
    createdAt: 2,
    content: [
      { type: "text", text: "ran it" },
      {
        type: "tool-call",
        toolCallId: "call-1",
        toolName: "python",
        result: {
          text: "wrote report.csv",
          images: [],
          sessionId,
          files: [{ name: "report.csv", size: 8 }],
        },
      },
    ],
  } as unknown as Message;
}

test("a chat moved into a project still names the sandbox its files are in", () => {
  // Written while the chat was loose, so the id is the thread's, not project-<id>.
  const messages = [toolMessage("thread-1")];
  assert.equal(recordedSandboxSessionId(messages), "thread-1");
});

test("the most recent tool result wins when the scope changed mid-chat", () => {
  const messages = [toolMessage("thread-1"), toolMessage("project-p1")];
  assert.equal(recordedSandboxSessionId(messages), "project-p1");
});

test("a chat that never ran a tool records nothing, leaving the caller its default", () => {
  const messages = [
    {
      id: "m1",
      threadId: "thread-1",
      role: "user",
      createdAt: 1,
      content: [{ type: "text", text: "hello" }],
    } as unknown as Message,
  ];
  assert.equal(recordedSandboxSessionId(messages), undefined);
});

test("an mcp tool answering with its own session id is not read as a sandbox", () => {
  const messages = [
    {
      id: "m2",
      threadId: "thread-1",
      role: "assistant",
      createdAt: 2,
      content: [
        {
          type: "tool-call",
          toolCallId: "call-1",
          toolName: "search_docs",
          result: {
            text: "found 3 hits",
            images: [],
            sessionId: "someone-elses-session",
            files: [],
          },
        },
      ],
    } as unknown as Message,
  ];
  assert.equal(recordedSandboxSessionId(messages), undefined);
});

test("a python result missing the wrapper shape is not read as a sandbox", () => {
  const messages = [
    {
      id: "m2",
      threadId: "thread-1",
      role: "assistant",
      createdAt: 2,
      content: [
        {
          type: "tool-call",
          toolCallId: "call-1",
          toolName: "python",
          // No images/files: the app's own wrapper always carries both.
          result: { text: "ok", sessionId: "not-a-wrapper" },
        },
      ],
    } as unknown as Message,
  ];
  assert.equal(recordedSandboxSessionId(messages), undefined);
});

test("a result without a session id is skipped rather than read as one", () => {
  const messages = [
    {
      id: "m2",
      threadId: "thread-1",
      role: "assistant",
      createdAt: 2,
      content: [
        { type: "tool-call", toolCallId: "c", toolName: "python", result: "plain text" },
        {
          type: "tool-call",
          toolCallId: "d",
          toolName: "python",
          result: { text: "ok", images: [], sessionId: "", files: [] },
        },
      ],
    } as unknown as Message,
  ];
  assert.equal(recordedSandboxSessionId(messages), undefined);
});
