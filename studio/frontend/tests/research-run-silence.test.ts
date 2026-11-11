// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
Object.assign(globalThis.window, {
  setTimeout: globalThis.setTimeout,
  clearTimeout: globalThis.clearTimeout,
});

const { followResearchRun } = await import(
  "../src/features/chat/api/research-api.ts"
);

const RUN_ID = "run-silence";

function snapshot(overrides: Record<string, unknown> = {}) {
  return {
    id: RUN_ID,
    threadId: "thread-1",
    userMessageId: "user-1",
    status: "running",
    steps: [],
    sources: [],
    lastEventSeq: 0,
    createdAt: 1,
    updatedAt: 1,
    ...overrides,
  };
}

function frame(id: number, event: string, data: Record<string, unknown>) {
  return `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

test("a research stream that goes silent without closing is dropped and resumed", async () => {
  const original = globalThis.fetch;
  const encoder = new TextEncoder();
  const eventUrls: string[] = [];
  let deadStreamCancelled = false;
  const terminal = snapshot({ status: "completed", lastEventSeq: 2 });
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/events")) {
      eventUrls.push(url);
      const first = eventUrls.length === 1;
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            if (!first) {
              controller.enqueue(
                encoder.encode(
                  frame(2, "run.completed", { attempt: 0, run: terminal }),
                ),
              );
              controller.close();
              return;
            }
            controller.enqueue(
              encoder.encode(frame(1, "report.updated", { delta: 1 })),
            );
            init?.signal?.addEventListener("abort", () =>
              controller.error(init.signal?.reason),
            );
          },
          cancel() {
            if (first) deadStreamCancelled = true;
          },
        }),
        { status: 200, headers: { "content-type": "text/event-stream" } },
      );
    }
    return new Response(JSON.stringify(snapshot({ lastEventSeq: 1 })), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;
  try {
    const ids: number[] = [];
    let last: unknown;
    for await (const update of followResearchRun(RUN_ID, {
      initialRun: snapshot() as never,
      silenceTimeoutMs: 50,
      signal: AbortSignal.timeout(2_000),
    })) {
      if (update.event) ids.push(update.event.id);
      last = update.run;
    }
    assert.deepEqual(ids, [1, 2]);
    assert.equal((last as { status: string }).status, "completed");
    assert.equal(deadStreamCancelled, true);
    assert.match(eventUrls[1], /after=1$/);
  } finally {
    globalThis.fetch = original;
  }
});
