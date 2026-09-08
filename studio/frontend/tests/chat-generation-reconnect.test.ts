// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test, { afterEach } from "node:test";

import type { ChatGenerationRun } from "../src/features/chat/api/chat-generation-api.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
registerBundlerResolver();
installLocalStorageFake();

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const {
  cancelChatGenerationRun,
  chatGenerationStopPlan,
  createChatGenerationRun,
  createChatGenerationRunUntilAbort,
  explicitStopSignal,
  isLegacyFallbackChatGenerationAdmissionError,
  isToolEnabledChatGenerationAdmissionError,
  followChatGenerationRun,
  supportsChatGenerationRuns,
} = await import("../src/features/chat/api/chat-generation-api.ts");

test("durable admission ignores detach but still observes explicit Stop", () => {
  const detached = new AbortController();
  const detachedAdmission = explicitStopSignal(detached.signal);
  detached.abort({ detach: true });
  assert.equal(detachedAdmission.signal.aborted, false);
  detachedAdmission.dispose();

  const stopped = new AbortController();
  const stoppedAdmission = explicitStopSignal(stopped.signal);
  stopped.abort({ detach: false });
  assert.equal(stoppedAdmission.signal.aborted, true);
  stoppedAdmission.dispose();
});

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

const createInput = () => ({
  runId: "run-1",
  threadId: "thread-1",
  userMessageId: "user-1",
  assistantMessageId: "assistant-1",
  requestPayload: run("queued", 1).requestPayload,
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
  const body = new ReadableStream({
    start(controller) {
      for (const value of frames) controller.enqueue(encoder.encode(value));
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

test("reconnect resumes from the applied cursor without duplicate chunks", async () => {
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
      JSON.stringify(follows ? run("completed", 4) : run("queued", 2)),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      },
    );
  }) as typeof fetch;
  const sequences: number[] = [];
  const snapshots: Array<[string, number]> = [];
  for await (const update of followChatGenerationRun("run-1", {
    initialRun: run("running", 0),
    replayFrom: 0,
  })) {
    if (update.event) sequences.push(update.event.seq);
    else snapshots.push([update.run.status, sequences.length]);
  }
  assert.deepEqual(sequences, [1, 2, 3, 4]);
  assert.deepEqual(snapshots, [
    ["running", 0],
    ["completed", 2],
  ]);
  assert.match(eventUrls[0], /after=0$/);
  assert.match(eventUrls[1], /after=2$/);
  assert.equal(
    eventUrls.some((url) => url.includes("chat/completions")),
    false,
  );
});

test("durable replay normalizes reasoning summary control frames", async () => {
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    if (String(input).includes("/events")) {
      return sse([
        frame(1, "chunk", { type: "reasoning_summary", duration_ms: 3200 }),
        frame(2, "run.completed", { status: "completed" }, run("completed", 2)),
      ]);
    }
    return new Response(JSON.stringify(run("completed", 2)), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;
  const chunks: object[] = [];
  for await (const update of followChatGenerationRun("run-1", {
    initialRun: run("running", 0),
    replayFrom: 0,
  })) {
    if (update.event?.type === "chunk") chunks.push(update.event.payload);
  }
  assert.deepEqual(chunks, [{ _reasoningDurationMs: 3200 }]);
});

test("a backend without chat runs selects the legacy path", async () => {
  globalThis.fetch = (async () =>
    new Response(null, { status: 404 })) as typeof fetch;
  assert.equal(await supportsChatGenerationRuns("thread-1"), false);
});

test("tool-enabled durable admission errors select the legacy stream", async () => {
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify({ detail: "Tool-enabled chat runs use the legacy streaming path" }),
      { status: 400, headers: { "content-type": "application/json" } },
    )) as typeof fetch;
  await assert.rejects(
    createChatGenerationRun(createInput()),
    (error: unknown) => {
      assert.equal(isToolEnabledChatGenerationAdmissionError(error), true);
      return true;
    },
  );
});

test("credential-safe durable admission errors select the legacy stream", async () => {
  globalThis.fetch = (async () =>
    new Response(JSON.stringify({ detail: "Credentials cannot be persisted" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    })) as typeof fetch;
  await assert.rejects(
    createChatGenerationRun(createInput()),
    (error: unknown) => {
      assert.equal(isLegacyFallbackChatGenerationAdmissionError(error), true);
      return true;
    },
  );
});

test("missing history rows select the legacy stream", async () => {
  for (const [status, detail] of [
    [404, "Thread not found"],
    [400, "userMessageId must identify a user message in the thread"],
  ] as const) {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ detail }), {
        status,
        headers: { "content-type": "application/json" },
      })) as typeof fetch;
    await assert.rejects(
      createChatGenerationRun(createInput()),
      (error: unknown) => isLegacyFallbackChatGenerationAdmissionError(error),
    );
  }
});

test("an ambiguous create retries the same run instead of starting generation twice", async () => {
  const bodies: string[] = [];
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    bodies.push(String(init?.body));
    if (bodies.length === 1) throw new TypeError("network reset");
    return new Response(JSON.stringify(run("queued", 1)), {
      status: 202,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;
  const created = await createChatGenerationRun(createInput());
  assert.equal(created.id, "run-1");
  assert.equal(bodies.length, 2);
  assert.equal(JSON.parse(bodies[0]).runId, JSON.parse(bodies[1]).runId);
});

test("Stop cancels the server run while an event reconnect is delayed", async () => {
  const controller = new AbortController();
  let eventCalls = 0;
  let cancelCalls = 0;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/cancel")) {
      cancelCalls += 1;
      return new Response(JSON.stringify(run("cancelled", 0)), { status: 200 });
    }
    if (url.includes("/events")) {
      eventCalls += 1;
      throw new TypeError("offline");
    }
    return new Response(JSON.stringify(run("running", 0)), { status: 200 });
  }) as typeof fetch;
  const following = (async () => {
    for await (const update of followChatGenerationRun("run-1", {
      initialRun: run("running", 0),
      replayFrom: 0,
      signal: controller.signal,
    })) {
      assert.equal(update.run.status, "running");
    }
  })();
  while (eventCalls === 0)
    await new Promise((resolve) => setTimeout(resolve, 0));
  await cancelChatGenerationRun("run-1");
  controller.abort();
  await following;
  assert.equal(cancelCalls, 1);
  assert.equal(eventCalls, 1);
});

test("completed, cancelled, and backend-restarted snapshots are terminal", async () => {
  for (const status of ["completed", "cancelled", "failed"] as const) {
    const terminal = run(status, 0);
    globalThis.fetch = (async (input: RequestInfo | URL) =>
      String(input).includes("/events")
        ? sse([])
        : new Response(JSON.stringify(terminal), {
            status: 200,
          })) as typeof fetch;
    const seen: string[] = [];
    for await (const update of followChatGenerationRun("run-1", {
      initialRun: terminal,
      replayFrom: 0,
    })) {
      seen.push(update.run.status);
    }
    assert.deepEqual(seen, [status]);
  }
});

test("Stop during create cancels the run after its delayed reply", async () => {
  const controller = new AbortController();
  let releaseCreate!: () => void;
  const delayed = new Promise<void>((resolve) => {
    releaseCreate = resolve;
  });
  let cancelled = 0;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    if (String(input).endsWith("/cancel")) {
      cancelled += 1;
      return new Response(JSON.stringify(run("cancelled", 1)), { status: 200 });
    }
    await delayed;
    return new Response(JSON.stringify(run("queued", 1)), { status: 202 });
  }) as typeof fetch;
  const creating = createChatGenerationRunUntilAbort(
    createInput(),
    controller.signal,
  );
  controller.abort({ detach: false });
  assert.equal(await creating, null);
  releaseCreate();
  while (cancelled === 0)
    await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(cancelled, 1);
});

test("Stop before admission resolves still reaches the server", () => {
  // Admission resolves long after the abort listener is installed (model auto-load,
  // RAG, attachment upload, first history save). A Stop in that window has no run id
  // and the turn may still fall back to the legacy stream, so it has to send the
  // cancel_id POST the backend stashes for a generation that registers afterwards.
  assert.deepEqual(chatGenerationStopPlan("pending", null), {
    cancelRunId: null,
    postLegacyCancel: true,
  });
  assert.deepEqual(chatGenerationStopPlan("legacy", null), {
    cancelRunId: null,
    postLegacyCancel: true,
  });

  // Once the run exists, cancelling it is enough and is the precise thing to do.
  assert.deepEqual(chatGenerationStopPlan("pending", "run-7"), {
    cancelRunId: "run-7",
    postLegacyCancel: false,
  });
  assert.deepEqual(chatGenerationStopPlan("durable", "run-7"), {
    cancelRunId: "run-7",
    postLegacyCancel: false,
  });

  // Durable with no id means create was aborted before it replied; that path chains
  // its own cancel onto the pending create, so a second POST would be noise.
  assert.deepEqual(chatGenerationStopPlan("durable", null), {
    cancelRunId: null,
    postLegacyCancel: false,
  });
});
