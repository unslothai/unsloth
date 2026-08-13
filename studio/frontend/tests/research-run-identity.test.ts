// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #8483: every delta event minted a new run object, so every component selecting the run
// re-rendered ~12x/s for the whole synthesis - including ChatPage, which owns the thread pane.
// The backend omits the run from those events by design (_DELTA_ONLY_EVENTS in
// studio/backend/routes/research_runs.py); the frontend must not reinvent it.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
registerBundlerResolver();
installLocalStorageFake();

const { followResearchRun } = await import(
  "../src/features/chat/api/research-api.ts"
);

type AnyRecord = Record<string, unknown>;

const RUN_ID = "run-identity";

function snapshot(overrides: AnyRecord = {}): AnyRecord {
  return {
    id: RUN_ID,
    thread_id: "thread-1",
    user_message_id: "user-1",
    status: "running",
    plan: null,
    plan_revision: 0,
    plan_hash: null,
    steps: [],
    sources: [],
    last_event_seq: 0,
    created_at: 1,
    updated_at: 1,
    ...overrides,
  };
}

/** One SSE frame in the shape routes/research_runs.py writes. */
function frame(id: number, event: string, data: AnyRecord): string {
  return `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function sseResponse(frames: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of frames) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function stubFetch(frames: string[], finalSnapshot: AnyRecord): () => void {
  const original = globalThis.fetch;
  let streamed = false;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/events")) {
      if (streamed) {
        // A second connect would loop forever; hand back an empty stream instead.
        return sseResponse([]);
      }
      streamed = true;
      return sseResponse(frames);
    }
    return new Response(JSON.stringify(finalSnapshot), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof globalThis.fetch;
  return () => {
    globalThis.fetch = original;
  };
}

test("delta-only events reuse the run object they were given", async () => {
  const terminal = snapshot({
    status: "completed",
    last_event_seq: 4,
    updated_at: 9,
    report: "done",
  });
  const restore = stubFetch(
    [
      frame(1, "reasoning.updated", {
        created_at: 2,
        call_id: "call-1",
        reasoning_delta: "a",
      }),
      frame(2, "report.updated", { created_at: 3, length: 32, delta: 32 }),
      frame(3, "reasoning.updated", {
        created_at: 4,
        call_id: "call-1",
        reasoning_delta: "b",
      }),
      frame(4, "run.completed", { created_at: 9, attempt: 0, run: terminal }),
    ],
    terminal,
  );
  try {
    const runs: unknown[] = [];
    for await (const update of followResearchRun(RUN_ID, {
      initialRun: snapshot() as never,
    })) {
      runs.push(update.run);
    }
    // snapshot, three deltas, terminal.
    assert.equal(runs.length, 5);
    assert.equal(runs[1], runs[0], "first delta minted a new run object");
    assert.equal(runs[2], runs[0], "second delta minted a new run object");
    assert.equal(runs[3], runs[0], "third delta minted a new run object");
    assert.notEqual(
      runs[4],
      runs[0],
      "the terminal event carries its own snapshot and must replace the run",
    );
    assert.equal(
      (runs[4] as AnyRecord).status,
      "completed",
      "the terminal snapshot must still be applied",
    );
  } finally {
    restore();
  }
});

test("a run carried by an event still replaces the held run", async () => {
  const midway = snapshot({ status: "running", last_event_seq: 1, updated_at: 5 });
  const terminal = snapshot({ status: "completed", last_event_seq: 2, updated_at: 9 });
  const restore = stubFetch(
    [
      frame(1, "phase.started", { created_at: 5, phase: "synthesis", run: midway }),
      frame(2, "run.completed", { created_at: 9, attempt: 0, run: terminal }),
    ],
    terminal,
  );
  try {
    const runs: AnyRecord[] = [];
    for await (const update of followResearchRun(RUN_ID, {
      initialRun: snapshot() as never,
    })) {
      runs.push(update.run as unknown as AnyRecord);
    }
    assert.equal(runs.length, 3);
    assert.notEqual(runs[1], runs[0]);
    assert.equal(runs[1].updatedAt, 5);
    assert.equal(runs[2].status, "completed");
  } finally {
    restore();
  }
});
