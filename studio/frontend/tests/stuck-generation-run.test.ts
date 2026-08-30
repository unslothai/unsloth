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
  markServerActiveGenerationRunsUnknown,
  forgetServerActiveGenerationRun,
  threadHasDurableGenerationRun,
  claimLiveGenerationRun,
  releaseLiveGenerationRun,
  generationNeedsRecovery,
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
    /needsGenerationRecovery\s*=\s*generationUnsettled \|\|\s*\(generationUnfinished &&\s*generationIsCorroboratedLive\(custom, m\.threadId\)\)/,
    "an unfinished run needs corroboration; a completed-but-unsettled one must not, "
      + "because /chat-runs/active never lists a terminal run and its tail would be lost",
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

test("a failed second active-run read leaves the thread unanswered", () => {
  // Source-pinned for the same reason as the gate above. `missed` proves the first list
  // predates the run, so keeping it AND recording it as an answer would restore a live
  // reply as interrupted and re-enable a conflicting send.
  const provider = read("../src/features/chat/runtime-provider.tsx");
  assert.match(provider, /let answered = true;/);
  assert.match(
    provider,
    /catch \{[^}]*answered = false;/s,
    "a failed retry must not be promoted into an authoritative empty list",
  );
  assert.match(
    provider,
    /if \(answered\) \{\s*syncServerActiveGenerationRuns\(/,
    "the sync must be gated on the read having actually answered",
  );
});

test("the recovery follower settles when the follow throws a stall", () => {
  const provider = read("../src/features/chat/runtime-provider.tsx");
  assert.match(
    provider,
    /catch \(error\) \{\s*if \(!\(error instanceof ChatGenerationStalledError\)\) throw error;\s*followStalled = true;\s*\}/,
    "the deadline throw must reach the settlement block, not the outer no-op catch",
  );
  assert.match(
    provider,
    /if \(followStalled \|\| generationNeedsRecovery\(currentMetadata\)\) \{/,
    "a stall settles the reply even when the metadata already looks settled",
  );
  const start = provider.indexOf("for await (const update of followChatGenerationRun(");
  const settle = provider.indexOf("generationNeedsRecovery(currentMetadata)", start);
  const caught = provider.indexOf("instanceof ChatGenerationStalledError", start);
  assert.ok(start > 0 && caught > start && settle > caught, "the catch must precede the settle");
});

test("a failed refresh retracts this thread's earlier answer", () => {
  resetServerActiveGenerationRuns();
  syncServerActiveGenerationRuns("t-1", ["run-a"]);
  assert.equal(serverHasAnsweredActiveRuns("t-1"), true);
  markServerActiveGenerationRunsUnknown("t-1");
  assert.equal(
    serverHasAnsweredActiveRuns("t-1"),
    false,
    "an answer is a point in time; a failed refresh must retract it",
  );
  // And a run another tab started since must keep the benefit of the doubt.
  assert.equal(
    generationIsCorroboratedLive({ generationRunId: "run-b" }, "t-1"),
    true,
  );
});

test("a terminal run stops making its thread durable", () => {
  resetServerActiveGenerationRuns();
  syncServerActiveGenerationRuns("t-2", ["run-c"]);
  assert.equal(threadHasDurableGenerationRun("t-2"), true);
  forgetServerActiveGenerationRun("run-c");
  assert.equal(
    threadHasDurableGenerationRun("t-2"),
    false,
    "a subscriber-owned stream started next must not inherit the durable cap",
  );
});

test("a locally interrupted follower is not revived by the benefit of the doubt", () => {
  resetServerActiveGenerationRuns();
  const stalled = {
    generationRunId: "run-d",
    generationLocallyInterrupted: true,
  };
  // No answer for this thread, which is normally enough to keep a run running.
  assert.equal(generationIsCorroboratedLive(stalled, "t-3"), false);
  assert.equal(generationNeedsRecovery(stalled), false);
  // The server still naming it live overrules the marker.
  syncServerActiveGenerationRuns("t-3", ["run-d"]);
  assert.equal(generationIsCorroboratedLive(stalled, "t-3"), true);
});

test("the marker is cleared when the server corroborates the run", () => {
  const provider = read("../src/features/chat/runtime-provider.tsx");
  assert.match(
    provider,
    /generationLocallyInterrupted: false,/,
    "history.load must let the server overrule a follower that gave up locally",
  );
  assert.match(
    provider,
    /generationLocallyInterrupted: true,/,
    "the stall settlement must stamp the marker",
  );
  assert.match(
    provider,
    /markServerActiveGenerationRunsUnknown\(remoteId\)/,
    "a failed second read must retract this thread's answer",
  );
  assert.match(
    provider,
    /if \(isTerminalChatGenerationRun\(update\.run\)\) \{\s*\/\/[^]*?forgetServerActiveGenerationRun\(runId\)/,
    "a terminal run must leave the server-active map",
  );
});

test("a legacy fallback releases the durable claim at once", () => {
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  const fallback = adapter.indexOf('generationDecision = "legacy";\n                    //');
  assert.ok(fallback > 0, "the legacy-fallback branch moved");
  const release = adapter.indexOf("releaseLiveGenerationRun(cancelId);", fallback);
  const nextClaim = adapter.indexOf("claimLiveGenerationRun(", fallback);
  assert.ok(
    release > 0 && (nextClaim === -1 || release < nextClaim),
    "the pre-admission claim must go before the subscriber-owned stream starts",
  );
});

test("the initial durable stream marks itself interrupted when it stalls", () => {
  // The recovery follower is not the only one holding the deadline. The adapter runs its
  // own follower for the first stream of a turn, and when THAT one gives up the persisted
  // metadata still reads running and unsettled. Without the marker, generationNeedsRecovery
  // stays true, the next reload attaches another follower, and the composer is blocked for
  // another full deadline.
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  const stream = adapter.indexOf("const durableStream = async function* () {");
  assert.ok(stream > 0, "the durable stream moved");
  const caught = adapter.indexOf("instanceof ChatGenerationStalledError", stream);
  assert.ok(caught > 0, "the adapter's own follower must catch the stall");
  const marked = adapter.indexOf("generationStalled = true;", caught);
  assert.ok(marked > 0, "catching the stall without recording it changes nothing");
  // Recording it is only useful if it reaches the metadata that survives a reload.
  const custom = adapter.indexOf("const generationCustom = ()");
  assert.ok(custom > 0 && custom < stream, "the metadata builder moved");
  const persisted = adapter.indexOf(
    "generationLocallyInterrupted: generationStalled,",
    custom,
  );
  assert.ok(
    persisted > custom && persisted < stream,
    "the stall marker must be persisted alongside generationSettled",
  );
  // The marker alone only stops the next reload reviving the run. Without an incomplete
  // reason the final yield writes `incomplete: undefined`, assistant-ui reads the partial
  // reply as finished, and there is no Continue until a reload rebuilds the reason.
  const reason = adapter.indexOf('incompleteReason = "interrupted";', caught);
  assert.ok(
    reason > caught && reason < adapter.indexOf("if (generationStatus ===", caught),
    "a stalled initial stream must settle as interrupted, not as completed",
  );
});

test("a completed run overrides the local interruption marker", () => {
  resetServerActiveGenerationRuns();
  // The backend resumed and finished after the follower gave up. /chat-runs/active
  // excludes completed runs, so it can never clear the marker for this one: honouring
  // it here would leave the reply running forever with its event tail never imported.
  const finished = {
    generationRunId: "run-e",
    generationStatus: "completed",
    generationSettled: false,
    generationLocallyInterrupted: true,
  };
  assert.equal(generationNeedsRecovery(finished), true);
  assert.equal(generationIsCorroboratedLive(finished, "t-4"), true);
  // Still non-terminal, so the marker still holds.
  assert.equal(
    generationNeedsRecovery({ ...finished, generationStatus: "running" }),
    false,
  );
});

test("a failed initial active-run read retracts the earlier answer too", () => {
  const provider = read("../src/features/chat/runtime-provider.tsx");
  assert.match(
    provider,
    /if \(!activeGenerationRunsLoaded\) \{[^]*?markServerActiveGenerationRunsUnknown\(remoteId\)/,
    "the initial-read failure path must retract the stale answer, not only the retry",
  );
});

test("the lease is renewed while the model is being prepared", () => {
  const runs = read("../../backend/core/inference/chat_generation_runs.py");
  assert.match(runs, /_renew_lease_while_preparing/);
  assert.match(
    runs,
    /_PREPARE_RENEW_MAX_SECONDS = /,
    "the renewal must be bounded, or a load that never returns is kept alive forever",
  );
  assert.match(
    runs,
    /def _renew_interval_seconds\(\)/,
    "the cadence must derive from the configured lease, not a constant that can exceed it",
  );
  // The heartbeat has to span the lifecycle gate too: a run waiting on it is still
  // queued and ages from created_at with nothing renewing it.
  const guard = runs.indexOf("async with self._lease_heartbeat(run_id):");
  const gate = runs.indexOf("await activity.start(cancel_event)", guard);
  const produce = runs.indexOf("await produce_openai_chat_completions(", guard);
  assert.ok(
    guard > 0 && gate > guard && produce > gate,
    "the heartbeat must open before the gate wait and still cover preparation",
  );
});
test("retracting an answer also drops that thread's stale run mappings", () => {
  resetServerActiveGenerationRuns();
  syncServerActiveGenerationRuns("t-5", ["run-f"]);
  const stalled = {
    generationRunId: "run-f",
    generationStatus: "running",
    generationLocallyInterrupted: true,
  };
  // While the answer stands, the server's word wins and the message is live.
  assert.equal(generationIsCorroboratedLive(stalled, "t-5"), true);
  markServerActiveGenerationRunsUnknown("t-5");
  // Once retracted it must not keep winning from the leftover mapping: that pairs a
  // running message with generationNeedsRecovery=false, so nothing would ever settle it.
  assert.equal(generationIsCorroboratedLive(stalled, "t-5"), false);
  assert.equal(generationNeedsRecovery(stalled), false);
  assert.equal(threadHasDurableGenerationRun("t-5"), false);
});

test("keep-alive comments hold the follower open through a long preparation", async () => {
  // The stream stays open and sends only comments for the whole of a model load. Before
  // this, the loop never advanced, so the snapshot poll that would have seen the lease
  // move never ran, and a healthy run was reported interrupted at the deadline.
  const api = await import("../src/features/chat/api/chat-generation-api");
  let push: ((chunk: Uint8Array) => void) | undefined;
  let close: (() => void) | undefined;
  const encoder = new TextEncoder();
  const original = globalThis.fetch;
  globalThis.fetch = (async (_url: string, init?: RequestInit) => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        push = (chunk) => controller.enqueue(chunk);
        close = () => controller.close();
        init?.signal?.addEventListener("abort", () =>
          controller.error(new DOMException("aborted", "AbortError")),
        );
      },
    });
    return new Response(body, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  }) as typeof fetch;

  try {
    const seen: string[] = [];
    const run = { id: "r", status: "running", lastEventSeq: 0, updatedAt: 1 };
    const follow = (async () => {
      for await (const update of api.followChatGenerationRun("r", {
        initialRun: run as never,
        stallTimeoutMs: 200,
      })) {
        seen.push(update.source);
        if (update.run.status !== "running") break;
      }
    })();
    // Only comments, spaced so the deadline would have fired twice without them.
    for (let i = 0; i < 4; i += 1) {
      await new Promise((r) => setTimeout(r, 120));
      push?.(encoder.encode(": keep-alive\n\n"));
    }
    push?.(
      encoder.encode(
        `data: ${JSON.stringify({ seq: 1, type: "chunk", payload: {}, run: { ...run, status: "completed", lastEventSeq: 1 } })}\n\n`,
      ),
    );
    close?.();
    await follow;
    assert.ok(seen.length > 0, "the follower must have survived to see the event");
  } finally {
    globalThis.fetch = original;
  }
});

test("a run finished between the two reads never reaches the durable registry", () => {
  // The registry sync runs BEFORE the overlay, so skipping only at the overlay left the
  // thread reading as durable. In another tab nothing removes that mapping, and the next
  // subscriber-owned stream on the thread would be capped and lose its tail.
  const provider = read("../src/features/chat/runtime-provider.tsx");
  const filter = provider.indexOf("terminalMessageRuns");
  const sync = provider.indexOf("syncServerActiveGenerationRuns(", filter);
  assert.ok(filter > 0, "the terminal-message filter is missing");
  assert.ok(sync > filter, "the filter must run before the registry sync");
  assert.match(
    provider,
    /activeGenerationRuns = activeGenerationRuns\.filter\(\s*\(run\) => !terminalMessageRuns\.has\(run\.assistantMessageId\),\s*\);/,
    "the list itself must be filtered, so sync and overlay both see it",
  );
});


test("a pre-admission claim does not make the thread bounded yet", () => {
  resetServerActiveGenerationRuns();
  // The claim is taken before the create POST so a recovery trigger during that await
  // cannot start a second follower. It must not also cap the checkpoints: until the POST
  // lands there is no server-side run, so those checkpoints are the only persistence, and
  // createChatGenerationRun retries transient failures until aborted, so the await can
  // outlast the 30 minute cap. A capped thread is dropped from the schedule for good.
  claimLiveGenerationRun("run-1", "thread-1", { provisional: true });
  assert.equal(threadHasDurableGenerationRun("thread-1"), false);

  // Admission succeeded: the second, non-provisional claim confirms it.
  claimLiveGenerationRun("run-1", "thread-1");
  assert.equal(threadHasDurableGenerationRun("thread-1"), true);
  releaseLiveGenerationRun("run-1");
  assert.equal(threadHasDurableGenerationRun("thread-1"), false);
});

test("releasing a provisional claim clears it rather than leaving it bounded", () => {
  resetServerActiveGenerationRuns();
  // The legacy fallback path releases without ever admitting. A stale provisional entry
  // would then make the NEXT claim of the same id read as unbounded.
  claimLiveGenerationRun("run-2", "thread-2", { provisional: true });
  releaseLiveGenerationRun("run-2");
  claimLiveGenerationRun("run-2", "thread-2");
  assert.equal(threadHasDurableGenerationRun("thread-2"), true);
  releaseLiveGenerationRun("run-2");
});
