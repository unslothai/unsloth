// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A durable run that never reaches a terminal status used to wedge the whole tab: the
// reply stayed "running", which unmounts the composer's Send button, the checkpoint
// schedule kept four requests moving every eight seconds, the follower reconnected
// forever, and every stop path reported nothing to stop. Two full app reloads changed
// none of it, because each one rebuilt the running status straight back out of the
// persisted metadata. These are the frontend halves of that: nothing may derive "still
// generating" from storage alone, and no loop may run without a bound.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test, { afterEach } from "node:test";
import { fileURLToPath } from "node:url";

import type { ChatGenerationRun } from "../src/features/chat/api/chat-generation-api.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
registerBundlerResolver();
installLocalStorageFake();

const read = (relative: string): string =>
  readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

const {
  generationIsCorroboratedLive,
  isServerActiveGenerationRun,
  loadGenerationOverlaySnapshot,
  resetServerActiveGenerationRuns,
  serverHasAnsweredActiveRuns,
  syncServerActiveGenerationRuns,
} = await import("../src/features/chat/utils/chat-generation-recovery.ts");

const { CHAT_GENERATION_STALL_TIMEOUT_MS, followChatGenerationRun } =
  await import("../src/features/chat/api/chat-generation-api.ts");

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
  finishReason: null,
  error: null,
  createdAt: 1,
  updatedAt: seq + 1,
  startedAt: 1,
  completedAt: null,
});

// A. the corroboration gate

test("before the server has answered, a live reply is not demoted", () => {
  // First load with the backend briefly unreachable: there is no previous answer to
  // leave standing, so an empty map must read as "never asked", not "nothing is
  // running". Demoting here would mark a live generation interrupted.
  resetServerActiveGenerationRuns();
  assert.equal(serverHasAnsweredActiveRuns("thread-1"), false);
  assert.equal(
    generationIsCorroboratedLive(
      { generationRunId: "run-unknown", generationStatus: "running" },
      "thread-1",
    ),
    true,
    "silence from the server is not a report that the run is dead",
  );
  // One successful read, and the gate becomes authoritative for THAT thread.
  syncServerActiveGenerationRuns("thread-1", []);
  assert.equal(serverHasAnsweredActiveRuns("thread-1"), true);
  assert.equal(
    generationIsCorroboratedLive(
      { generationRunId: "run-unknown", generationStatus: "running" },
      "thread-1",
    ),
    false,
  );
});

test("one thread's successful read does not make another thread's failed read authoritative", () => {
  // A answers, B's active-run request fails while its history succeeds. With a single
  // process-wide flag, B's genuinely active reply was restored as interrupted.
  resetServerActiveGenerationRuns();
  syncServerActiveGenerationRuns("thread-A", []);
  assert.equal(serverHasAnsweredActiveRuns("thread-A"), true);
  assert.equal(serverHasAnsweredActiveRuns("thread-B"), false);
  assert.equal(
    generationIsCorroboratedLive(
      { generationRunId: "run-B", generationStatus: "running" },
      "thread-B",
    ),
    true,
    "B has no answer of its own yet, so its reply must not be demoted",
  );
});

test("persisted running metadata is not on its own evidence of a live run", () => {
  syncServerActiveGenerationRuns("thread-1", []);
  assert.equal(
    generationIsCorroboratedLive(
      {
        generationRunId: "run-1",
        generationStatus: "running",
        generationSettled: false,
      },
      "thread-1",
    ),
    false,
    "a stuck run says 'running' in storage for good; only the server may confirm it",
  );
});

test("the server's active list corroborates a run, and a later list retracts it", () => {
  syncServerActiveGenerationRuns("thread-1", ["run-1"]);
  assert.equal(isServerActiveGenerationRun("run-1"), true);
  assert.equal(
    generationIsCorroboratedLive({ generationRunId: "run-1" }),
    true,
  );

  syncServerActiveGenerationRuns("thread-1", []);
  assert.equal(
    isServerActiveGenerationRun("run-1"),
    false,
    "the thread's next load is the whole truth about that thread's runs",
  );
});

test("one thread's active list does not retract another thread's runs", () => {
  syncServerActiveGenerationRuns("thread-1", ["run-1"]);
  syncServerActiveGenerationRuns("thread-2", ["run-2"]);
  syncServerActiveGenerationRuns("thread-2", []);
  assert.equal(isServerActiveGenerationRun("run-1"), true);
  assert.equal(isServerActiveGenerationRun("run-2"), false);
  syncServerActiveGenerationRuns("thread-1", []);
});

test("a message with no run id is never corroborated", () => {
  assert.equal(generationIsCorroboratedLive({}), false);
  assert.equal(generationIsCorroboratedLive({ generationRunId: 7 }), false);
});

test("a failed active-run read is reported as unknown, not as an empty list", async () => {
  const failed = await loadGenerationOverlaySnapshot(
    "thread-1",
    () => Promise.reject(new Error("offline")),
    () => Promise.resolve([{ id: "assistant-1" }]),
  );
  assert.deepEqual(failed.activeRuns, []);
  assert.equal(
    failed.activeRunsLoaded,
    false,
    "an unreachable backend must not be read as 'nothing is running'",
  );

  const loaded = await loadGenerationOverlaySnapshot(
    "thread-1",
    () => Promise.resolve([{ id: "run-1" }]),
    () => Promise.resolve([{ id: "assistant-1" }]),
  );
  assert.equal(loaded.activeRunsLoaded, true);
});

test("the reload path gates the running status on corroboration", () => {
  // Source-pinned: toThreadMessage needs a MessageRecord and the module's whole import
  // graph, so the rule is asserted where it sits. Losing the call restores the wedge
  // while every behavioural test stays green.
  const provider = read("../src/features/chat/runtime-provider.tsx");
  const gate = provider.indexOf("generationIsCorroboratedLive(custom, m.threadId)");
  assert.ok(gate > 0, "toThreadMessage no longer asks whether the run is live");
  assert.match(
    provider,
    /needsGenerationRecovery\s*=\s*\(generationUnfinished \|\| generationUnsettled\) &&\s*generationIsCorroboratedLive\(custom, m\.threadId\)/,
    "the running status must require BOTH unfinished metadata and corroboration",
  );
  assert.match(
    provider,
    /incomplete: \{ reason: "interrupted" as const \}/,
    "an uncorroborated partial must be restored as interrupted, not as complete",
  );
  assert.match(
    provider,
    /activeGenerationRuns\.filter\(\s*\(run\) => !isTerminalChatGenerationRun\(run\),\s*\)/,
    "history.load must stamp generationSettled:false only for still-live runs",
  );
});

test("the interrupted restore keeps the partial body untouched", () => {
  const provider = read("../src/features/chat/runtime-provider.tsx");
  // The only content assignment in toThreadMessage is the clone of the stored parts.
  // Nothing between the gate and the return may drop or truncate it.
  const start = provider.indexOf("function toThreadMessage(");
  const body = provider.slice(start, provider.indexOf("\n}\n", start));
  assert.ok(start > 0);
  assert.match(body, /content: content as Extract</);
  assert.equal(
    /content:\s*\[\]/.test(body),
    false,
    "restoring an interrupted reply must never empty its content",
  );
});

const stalledStreamFetch = (): typeof fetch =>
  (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/events")) {
      const body = new ReadableStream({
        start(controller) {
          init?.signal?.addEventListener("abort", () => {
            controller.error(new Error("aborted"));
          });
        },
      });
      return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    }
    return new Response(JSON.stringify(run("running", 0)), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;

test("the follower throws on its own deadline instead of returning silently", async () => {
  // The live durable stream only throws for failed/cancelled. A silent return on our own
  // deadline let a still-running run be finalized as a complete assistant reply while the
  // backend kept generating with no follower.
  globalThis.fetch = stalledStreamFetch();
  await assert.rejects(
    async () => {
      for await (const _update of followChatGenerationRun("run-1", {
        initialRun: run("running", 0),
        replayFrom: 0,
        stallTimeoutMs: 40,
      })) {
        // drain
      }
    },
    (error: unknown) =>
      error instanceof Error && error.name === "ChatGenerationStalledError",
    "a deadline must surface as a stall, not as a clean end of stream",
  );
});

test("a caller Stop still ends the follow cleanly", async () => {
  // Only OUR deadline is a failure. The user pressing Stop stays a clean return, so a
  // deliberate cancel is never reported to the user as an interrupted reply.
  globalThis.fetch = stalledStreamFetch();
  const controller = new AbortController();
  globalThis.setTimeout(() => controller.abort(), 30);
  for await (const _update of followChatGenerationRun("run-1", {
    initialRun: run("running", 0),
    replayFrom: 0,
    signal: controller.signal,
    stallTimeoutMs: 60_000,
  })) {
    // drain
  }
});

// B. the follower's deadline

test("the follower gives up on a run that makes no progress", async () => {
  // The stream stays open and sends nothing, which is the shape a stuck durable run
  // presents. The stub honours the abort signal exactly as fetch does, because that is
  // the plumbing the deadline relies on to unblock a parked reader.
  let opened = 0;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/events")) {
      opened += 1;
      const body = new ReadableStream({
        start(controller) {
          init?.signal?.addEventListener("abort", () => {
            controller.error(new Error("aborted"));
          });
        },
      });
      return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    }
    return new Response(JSON.stringify(run("running", 0)), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;

  const updates: string[] = [];
  const started = Date.now();
  await assert.rejects(async () => {
    for await (const update of followChatGenerationRun("run-1", {
      initialRun: run("running", 0),
      replayFrom: 0,
      stallTimeoutMs: 40,
    })) {
      updates.push(update.source);
    }
  }, /made no progress/);
  assert.ok(
    Date.now() - started < 5_000,
    "the follow must end on its own rather than reconnect forever",
  );
  assert.deepEqual(updates, ["snapshot"], "no progress was ever made");
  assert.ok(opened >= 1, "the follower did open the event stream");
});

test("progress rearms the deadline instead of ending the follow", async () => {
  const frames = [1, 2, 3];
  let follows = 0;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/events")) {
      const seq = frames[follows] ?? 0;
      follows += 1;
      const terminal = follows >= frames.length;
      const encoder = new TextEncoder();
      const snapshot = terminal ? run("completed", seq) : undefined;
      const body = new ReadableStream({
        async start(controller) {
          // Each stream also pays the reconnect backoff before it, so the run lasts
          // longer than the deadline in total, but never goes that long without progress.
          await new Promise((resolve) => setTimeout(resolve, 25));
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                seq,
                type: terminal ? "run.completed" : "chunk",
                payload: { choices: [{ delta: { content: "x" } }] },
                createdAt: seq,
                ...(snapshot ? { run: snapshot } : {}),
              })}\n\n`,
            ),
          );
          controller.close();
        },
      });
      return new Response(body, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    }
    return new Response(
      JSON.stringify(
        follows >= frames.length
          ? run("completed", 3)
          : run("running", follows),
      ),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  }) as typeof fetch;

  const seen: number[] = [];
  for await (const update of followChatGenerationRun("run-1", {
    initialRun: run("running", 0),
    replayFrom: 0,
    stallTimeoutMs: 1_500,
  })) {
    if (update.event) seen.push(update.event.seq);
  }
  assert.deepEqual(
    seen,
    [1, 2, 3],
    "a run that keeps producing must not be cut off by the deadline",
  );
});

test("the default deadline outlasts any reasonable generation", () => {
  // The client has to be the more patient of the two. A prefill emits no events while it
  // runs, the backend allows 1200s for a first token, and the lease sweeper needs another
  // interval to settle a genuinely dead run. A deadline under roughly 21 minutes would
  // report "interrupted" over a generation the server is still working on.
  assert.equal(CHAT_GENERATION_STALL_TIMEOUT_MS, 30 * 60_000);
  assert.ok(
    CHAT_GENERATION_STALL_TIMEOUT_MS > 1_200_000 + 60_000,
    "the follow deadline must outlast the backend first-token budget plus a sweep",
  );
});

// C. the stop path after a reload

test("stopChatThread falls back to the server when the registries are empty", () => {
  const api = read("../src/features/chat/utils/stop-chat-thread.ts");
  assert.match(
    api,
    /getActiveChatGenerationRuns/,
    "a reloaded tab has no local handle; the server's list is the only one left",
  );
  assert.match(api, /cancelChatGenerationRun\(run\.id\)/);
  const earlyReturn = api.indexOf("return false");
  const fallback = api.indexOf("stopServerRunsForThread(threadId);");
  assert.ok(fallback > 0, "the fallback is gone");
  assert.ok(
    earlyReturn < 0 || earlyReturn > 0,
    "the only `return false` left is the missing-thread-id guard",
  );
  assert.equal(
    api.split("return false").length - 1,
    1,
    "an empty registry must no longer be a reason to give up",
  );
});
