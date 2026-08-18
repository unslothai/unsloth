// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type { ChatGenerationRun } from "../src/features/chat/api/chat-generation-api.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
registerBundlerResolver();
installLocalStorageFake();

const {
  createChatGenerationRun,
  followChatGenerationRun,
  supportsChatGenerationRuns,
} = await import("../src/features/chat/api/chat-generation-api.ts");

const run = (
  status: ChatGenerationRun["status"],
  seq: number,
): ChatGenerationRun => ({
  id: "run-1",
  threadId: "thread-1",
  userMessageId: "user-1",
  assistantMessageId: "assistant-1",
  requestHash: "hash",
  requestPayload: { model: "local", messages: [], stream: true, max_tokens: 8 },
  status,
  cancelRequested: false,
  lastEventSeq: seq,
  finishReason: status === "completed" ? "stop" : null,
  error: null,
  createdAt: 1,
  updatedAt: seq + 1,
  startedAt: 1,
  completedAt: status === "completed" ? 9 : null,
});

const frame = (
  seq: number,
  type: string,
  payload: object,
  snapshot?: ChatGenerationRun,
) =>
  `id: ${seq}\nevent: ${type}\ndata: ${JSON.stringify({
    seq,
    type,
    payload,
    createdAt: seq,
    ...(snapshot ? { run: snapshot } : {}),
  })}\n\n`;

function sse(frames: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const value of frames) controller.enqueue(encoder.encode(value));
        controller.close();
      },
    }),
    { status: 200, headers: { "content-type": "text/event-stream" } },
  );
}

test("reconnect resumes from the applied cursor without duplicate chunks", async () => {
  const original = globalThis.fetch;
  const eventUrls: string[] = [];
  let follows = 0;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/events")) {
      eventUrls.push(url);
      follows += 1;
      return follows === 1
        ? sse([
            frame(1, "chunk", { choices: [{ delta: { content: "A" } }] }),
            frame(2, "chunk", { choices: [{ delta: { content: "B" } }] }),
          ])
        : sse([
            frame(3, "chunk", { choices: [{ delta: { content: "C" } }] }),
            frame(
              4,
              "run.completed",
              { status: "completed" },
              run("completed", 4),
            ),
          ]);
    }
    return new Response(
      JSON.stringify(run(follows ? "running" : "queued", 2)),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      },
    );
  }) as typeof fetch;
  try {
    const sequences: number[] = [];
    for await (const update of followChatGenerationRun("run-1", {
      initialRun: run("running", 0),
      replayFrom: 0,
    })) {
      if (update.event) sequences.push(update.event.seq);
    }
    assert.deepEqual(sequences, [1, 2, 3, 4]);
    assert.match(eventUrls[0], /after=0$/);
    assert.match(eventUrls[1], /after=2$/);
    assert.equal(
      eventUrls.some((url) => url.includes("chat/completions")),
      false,
    );
  } finally {
    globalThis.fetch = original;
  }
});

test("a backend without chat runs selects the legacy path", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(null, { status: 404 })) as typeof fetch;
  try {
    assert.equal(await supportsChatGenerationRuns("thread-1"), false);
  } finally {
    globalThis.fetch = original;
  }
});

test("an ambiguous create retries the same run instead of starting generation twice", async () => {
  const original = globalThis.fetch;
  const bodies: string[] = [];
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    bodies.push(String(init?.body));
    if (bodies.length === 1) throw new TypeError("network reset");
    return new Response(JSON.stringify(run("queued", 1)), {
      status: 202,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;
  try {
    const created = await createChatGenerationRun({
      runId: "run-1",
      threadId: "thread-1",
      userMessageId: "user-1",
      assistantMessageId: "assistant-1",
      requestPayload: run("queued", 1).requestPayload,
    });
    assert.equal(created.id, "run-1");
    assert.equal(bodies.length, 2);
    assert.equal(JSON.parse(bodies[0]).runId, JSON.parse(bodies[1]).runId);
  } finally {
    globalThis.fetch = original;
  }
});
