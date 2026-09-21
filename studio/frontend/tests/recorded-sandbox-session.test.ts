// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { allRecordedSandboxSessionIds } = await import(
  "../src/features/chat/utils/recorded-sandbox-session.ts"
);

type Message = Parameters<typeof allRecordedSandboxSessionIds>[0][number];

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
  assert.deepEqual(allRecordedSandboxSessionIds(messages), ["thread-1"]);
});

test("a scope change mid-chat leaves files in both folders, so both are named", () => {
  // One thread, two sandboxes: it ran a tool loose, joined a project, and ran
  // another. Answering with the newest alone would strand the first folder,
  // and its caller would think the chat had a single session id to hand out.
  const messages = [toolMessage("thread-1"), toolMessage("project-p1")];
  assert.deepEqual(allRecordedSandboxSessionIds(messages), [
    "project-p1",
    "thread-1",
  ]);
});

test("a chat that stayed put names its one sandbox once, however many tools ran", () => {
  const messages = [
    toolMessage("project-p1"),
    toolMessage("project-p1"),
    toolMessage("project-p1"),
  ];
  assert.deepEqual(allRecordedSandboxSessionIds(messages), ["project-p1"]);
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
  assert.deepEqual(allRecordedSandboxSessionIds(messages), []);
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
  assert.deepEqual(allRecordedSandboxSessionIds(messages), []);
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
  assert.deepEqual(allRecordedSandboxSessionIds(messages), []);
});

test("a result without a session id is skipped rather than read as one", () => {
  const messages = [
    {
      id: "m2",
      threadId: "thread-1",
      role: "assistant",
      createdAt: 2,
      content: [
        {
          type: "tool-call",
          toolCallId: "c",
          toolName: "python",
          result: "plain text",
        },
        {
          type: "tool-call",
          toolCallId: "d",
          toolName: "python",
          result: { text: "ok", images: [], sessionId: "", files: [] },
        },
      ],
    } as unknown as Message,
  ];
  assert.deepEqual(allRecordedSandboxSessionIds(messages), []);
});

test("a run that created no files still names the sandbox it ran in", () => {
  // No __FILES__ envelope is emitted when the call changed nothing AND when a
  // concurrent call shared the directory, so a run that DID write files can
  // arrive bare. The adapter wraps python and terminal results either way.
  const messages = [
    {
      id: "m2",
      threadId: "t",
      role: "assistant",
      createdAt: 2,
      content: [
        {
          type: "tool-call",
          toolCallId: "c",
          toolName: "python",
          result: {
            text: "42\n",
            images: [],
            sessionId: "project-p1",
            files: [],
          },
        },
      ],
    } as unknown as Message,
  ];
  assert.deepEqual(allRecordedSandboxSessionIds(messages), ["project-p1"]);
});

test("content that is not an array is skipped rather than thrown on", () => {
  // Legacy and imported chats carry a plain string here, and this runs from a
  // menu handler: throwing would take the click, not just the answer.
  const messages = [
    { id: "m1", threadId: "t", role: "user", createdAt: 1, content: "hello" },
    { id: "m2", threadId: "t", role: "assistant", createdAt: 2, content: null },
    { id: "m3", threadId: "t", role: "assistant", createdAt: 3 },
  ] as unknown as Message[];
  assert.deepEqual(allRecordedSandboxSessionIds(messages), []);
});

test("a null part inside a real content array does not stop the scan", () => {
  const messages = [
    {
      id: "m2",
      threadId: "t",
      role: "assistant",
      createdAt: 2,
      content: [
        null,
        {
          type: "tool-call",
          toolCallId: "c",
          toolName: "terminal",
          result: { text: "ok", images: [], sessionId: "thread-1", files: [] },
        },
        null,
      ],
    } as unknown as Message,
  ];
  assert.deepEqual(allRecordedSandboxSessionIds(messages), ["thread-1"]);
});

test("a tool-result shaped part is not mistaken for a tool call", () => {
  // Only "tool-call" parts carry the result this reads. A part typed
  // "tool-result" is a different shape and must not be half-read.
  const messages = [
    {
      id: "m2",
      threadId: "t",
      role: "assistant",
      createdAt: 2,
      content: [
        {
          type: "tool-result",
          toolName: "python",
          result: { text: "ok", images: [], sessionId: "thread-1", files: [] },
        },
      ],
    } as unknown as Message,
  ];
  assert.deepEqual(allRecordedSandboxSessionIds(messages), []);
});

test("a long history is walked in full and still answers inside a menu click", () => {
  // Naming every folder means the whole history, not an early return on the
  // newest. Runs on a menu click over a chat of thousands of turns, so the
  // bound stands: it is one pass over parts already in memory.
  const messages: Message[] = [];
  for (let i = 0; i < 5000; i += 1) {
    messages.push({
      id: `m${i}`,
      threadId: "t",
      role: "user",
      createdAt: i,
      content: [{ type: "text", text: "x" }],
    } as unknown as Message);
  }
  messages.push(toolMessage("newest-session"));
  const started = performance.now();
  assert.deepEqual(allRecordedSandboxSessionIds(messages), ["newest-session"]);
  assert.ok(
    performance.now() - started < 50,
    "a full pass has to stay inside a click",
  );
});
