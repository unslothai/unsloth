// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

register("./helpers/vite-env-loader.mjs", import.meta.url);
registerBundlerResolver();
installLocalStorageFake();
// The store registers a session-cleared listener at import.
Object.assign(globalThis.window as object, { addEventListener: () => {} });

const {
  ingestResearchUpdate,
  researchPhaseTitle,
  runningResearchActivityTitle,
  resetResearchRunState,
  researchProgressSummary,
  stepResultDetail,
  useResearchRunStore,
  watchResearchRun,
} = await import("../src/features/chat/stores/research-run-store.ts");

type AnyRecord = Record<string, unknown>;

const RUN_ID = "run-1";

function run(overrides: AnyRecord = {}): AnyRecord {
  return {
    id: RUN_ID,
    threadId: "thread-1",
    userMessageId: "user-1",
    status: "planning",
    plan: null,
    planRevision: 0,
    planHash: null,
    steps: [],
    sources: [],
    documentSources: [],
    lastEventSeq: 0,
    createdAt: 1,
    updatedAt: 1,
    retryCount: 0,
    ...overrides,
  };
}

function event(
  id: number,
  name: string,
  data: AnyRecord = {},
  snapshot: AnyRecord = run(),
): AnyRecord {
  return {
    id,
    event: name,
    createdAt: id,
    data: { ...data, run: snapshot },
    run: snapshot,
  };
}

// Clears the coalescing timers too, so one case's pending delta cannot land in the next.
function reset(): void {
  resetResearchRunState();
}

function activities() {
  return useResearchRunStore.getState().sessions[RUN_ID]?.activities ?? [];
}

// STREAM_EVENT_FLUSH_MS in the store, which does not export it.
function flushCoalescedDeltas(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 120));
}

// biome-ignore lint/suspicious/noExplicitAny: the store's event shape is exercised structurally
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ingest = ingestResearchUpdate as any;

test("a planning phase produces a live row before the plan exists", () => {
  reset();
  ingest(run());
  ingest(run(), event(1, "run.created", { status: "planning" }));
  ingest(
    run(),
    event(2, "phase.started", { phase: "planning", callId: "call-a" }),
  );

  const running = activities().filter((a) => a.state === "running");
  assert.equal(running.length, 1);
  assert.equal(running[0].title, "Planning an approach");
  assert.equal(runningResearchActivityTitle(activities()), "Planning an approach");
});

test("reasoning for a call lands on that call's phase row, not a second one", async () => {
  reset();
  ingest(run());
  ingest(
    run(),
    event(1, "phase.started", { phase: "planning", callId: "call-a" }),
  );
  ingest(
    run(),
    event(2, "reasoning.updated", {
      phase: "planning",
      callId: "call-a",
      reasoningDelta: "thinking",
    }),
  );
  await flushCoalescedDeltas();

  const rows = activities().filter((a) => a.kind === "reasoning");
  assert.equal(rows.length, 1);
  assert.equal(rows[0].reasoning, "thinking");
  assert.equal(rows[0].state, "running");
});

test("plan titles stream onto the planning row while the plan is still being written", () => {
  reset();
  ingest(run());
  ingest(run(), event(1, "phase.started", { phase: "planning", callId: "p" }));
  ingest(
    run(),
    event(2, "phase.progress", { phase: "planning", callId: "p", label: "Find the spec" }),
  );
  ingest(
    run(),
    event(3, "phase.progress", { phase: "planning", callId: "p", label: "Check adoption" }),
  );
  // A replayed duplicate must not double the list.
  ingest(
    run(),
    event(4, "phase.progress", { phase: "planning", callId: "p", label: "Check adoption" }),
  );

  const [row] = activities().filter((a) => a.kind === "reasoning");
  assert.deepEqual(row.previewLabels, ["Find the spec", "Check adoption"]);
  assert.equal(
    runningResearchActivityTitle(activities()),
    "Planning an approach · Check adoption",
  );
});

test("a progress label for an unknown call is ignored", () => {
  reset();
  ingest(run());
  ingest(
    run(),
    event(1, "phase.progress", { phase: "planning", callId: "gone", label: "Orphan" }),
  );

  assert.equal(activities().filter((a) => a.kind === "reasoning").length, 0);
});

test("phase.ended closes only its own call", () => {
  reset();
  ingest(run());
  ingest(run(), event(1, "phase.started", { phase: "decision", callId: "a" }));
  ingest(run(), event(2, "phase.ended", { phase: "decision", callId: "a" }));
  ingest(run(), event(3, "phase.started", { phase: "decision", callId: "b" }));

  const rows = activities().filter((a) => a.kind === "reasoning");
  assert.deepEqual(
    rows.map((row) => row.state),
    ["complete", "running"],
  );
});

// The old reducer closed the live reasoning row on any non-reasoning event.
test("an unrelated event does not close a live phase row", async () => {
  reset();
  ingest(run());
  ingest(run(), event(1, "phase.started", { phase: "synthesis", callId: "s" }));
  ingest(run(), event(2, "report.updated", { delta: "text" }));
  await flushCoalescedDeltas();

  const rows = activities().filter((a) => a.kind === "reasoning");
  assert.equal(rows.length, 1);
  assert.equal(rows[0].state, "running");
  // No duplicate "Writing the report" row alongside the synthesis phase row.
  assert.equal(activities().filter((a) => a.kind === "report").length, 0);
});

// Both rows are titled "Writing the report", so a closed synthesis row must still absorb the
// report deltas rather than let a second row appear beside it.
test("a legacy synthesis row absorbs the report instead of doubling", async () => {
  reset();
  ingest(run({ status: "running" }));
  ingest(
    run({ status: "running" }),
    event(1, "reasoning.updated", {
      phase: "synthesis",
      callId: "S",
      reasoningDelta: "t",
    }),
  );
  await flushCoalescedDeltas();
  ingest(run({ status: "running" }), event(2, "report.updated", { delta: "x" }));
  await flushCoalescedDeltas();

  assert.equal(activities().filter((a) => a.kind === "report").length, 0);
  assert.deepEqual(
    activities()
      .filter((a) => a.title === "Writing the report")
      .map((a) => a.kind),
    ["reasoning"],
  );
});

test("a run recorded before phase events still gets a report row", async () => {
  reset();
  ingest(run());
  ingest(run(), event(1, "report.updated", { delta: "text" }));
  await flushCoalescedDeltas();

  const rows = activities().filter((a) => a.kind === "report");
  assert.equal(rows.length, 1);
  assert.equal(rows[0].title, "Writing the report");
});

// Without the legacy close rule, a pre-phase-event run marked succeeded work failed.
test("a legacy run's reasoning rows still close without phase brackets", async () => {
  reset();
  ingest(run());
  ingest(
    run(),
    event(1, "reasoning.updated", {
      phase: "planning",
      callId: "A",
      reasoningDelta: "plan",
    }),
  );
  await flushCoalescedDeltas();
  ingest(run(), event(2, "plan.ready", { planRevision: 1 }));
  const failed = run({ status: "failed", lastEventSeq: 3 });
  ingest(failed, event(3, "run.failed", { error: "boom" }, failed));

  const [planning] = activities().filter((a) => a.kind === "reasoning");
  assert.equal(planning.state, "complete");
});

// The released loop writes no event between the audit and synthesis calls.
test("consecutive legacy calls close each other, leaving one spinner", async () => {
  reset();
  ingest(run({ status: "running" }));
  for (const [id, phase, callId] of [
    [1, "decision", "B"],
    [2, "synthesis_audit", "C"],
    [3, "synthesis", "D"],
  ] as const) {
    ingest(
      run({ status: "running" }),
      event(id, "reasoning.updated", { phase, callId, reasoningDelta: "x" }),
    );
    await flushCoalescedDeltas();
  }

  const running = activities().filter((a) => a.state === "running");
  assert.deepEqual(
    running.map((a) => a.title),
    ["Writing the report"],
  );

  const failed = run({ status: "failed", lastEventSeq: 4 });
  ingest(failed, event(4, "run.failed", { error: "boom" }, failed));
  const reasoningStates = activities()
    .filter((a) => a.kind === "reasoning")
    .map((a) => a.state);
  assert.deepEqual(reasoningStates, ["complete", "complete", "complete"]);
});

test("a bracketed row is not closed early by an unrelated event", () => {
  reset();
  ingest(run());
  ingest(run(), event(1, "phase.started", { phase: "synthesis", callId: "s" }));
  ingest(run(), event(2, "step.started", { stepPosition: 0, title: "Search" }));

  const [row] = activities().filter((a) => a.kind === "reasoning");
  assert.equal(row.state, "running");
});

test("a terminal run closes every row left running", () => {
  reset();
  ingest(run());
  ingest(run(), event(1, "phase.started", { phase: "planning", callId: "a" }));
  const cancelled = run({ status: "cancelled", lastEventSeq: 2 });
  ingest(cancelled, event(2, "run.cancelled", {}, cancelled));

  assert.equal(activities().every((a) => a.state !== "running"), true);
});

test("phase titles cover every phase the backend emits", () => {
  assert.equal(researchPhaseTitle("planning"), "Planning an approach");
  assert.equal(researchPhaseTitle("decision"), "Choosing the next step");
  assert.equal(researchPhaseTitle("synthesis_audit"), "Checking the evidence");
  assert.equal(researchPhaseTitle("synthesis"), "Writing the report");
  assert.equal(researchPhaseTitle("synthesis_recovery"), "Writing the report");
  assert.equal(researchPhaseTitle(undefined), "Working");
});

test("the header summary never reads 0 sources or 0 actions", () => {
  const summary = (overrides: AnyRecord) =>
    researchProgressSummary(run(overrides) as never, "12s");

  assert.equal(summary({ status: "planning" }), "12s · building the plan");
  assert.equal(
    summary({ status: "awaiting_approval" }),
    "12s · waiting for you",
  );
  assert.equal(
    summary({
      status: "running",
      steps: [
        { position: 0, title: "a", query: "q", status: "completed" },
        { position: 1, title: "b", query: "q", status: "running" },
      ],
      sources: [],
    }),
    // Names the step actually running; run.steps is not the plan, so there is no denominator.
    "12s · step 2",
  );
  assert.equal(
    summary({
      status: "running",
      steps: [{ position: 0, title: "a", query: "q", status: "running" }],
    }),
    "12s · step 1",
  );
  assert.equal(
    summary({
      status: "running",
      sources: [{ url: "https://e.com", title: "e" }],
      steps: [
        { position: 0, title: "a", query: "q", status: "completed" },
        { position: 1, title: "b", query: "q", status: "running" },
      ],
    }),
    "12s · 1 source · step 2",
  );
  // Document-only evidence still counts, and a failed step counts as finished.
  assert.equal(
    summary({
      status: "running",
      documentSources: [{ filename: "a.pdf", documentId: "d1" }],
      steps: [{ position: 0, title: "a", query: "q", status: "failed" }],
    }),
    "12s · 1 source · 1 step",
  );
});

test("an empty search reads as no results rather than a count of zero", () => {
  assert.equal(stepResultDetail(0), "No usable results");
  assert.equal(stepResultDetail(1), "1 source found");
  assert.equal(stepResultDetail(4), "4 sources found");
  // A fetch reads one page and never collects sources, so a zero count is not a bad outcome.
  assert.equal(stepResultDetail(0, "fetch"), "Page read");
  assert.equal(stepResultDetail(0, "search"), "No usable results");
});

// Ingestion used to run off this generator, so a stalled consumer froze the card.
test("a stalled watcher does not stop events reaching the store", async () => {
  reset();
  ingest(run());
  const iterator = watchResearchRun(RUN_ID)[Symbol.asyncIterator]();
  await iterator.next();

  const planned = run({
    status: "awaiting_approval",
    plan: { title: "Plan", steps: [{ title: "One", query: "q" }] },
    lastEventSeq: 1,
  });
  ingest(planned, event(1, "plan.ready", { planRevision: 1 }, planned));

  assert.equal(
    useResearchRunStore.getState().sessions[RUN_ID].run.status,
    "awaiting_approval",
  );
  const next = await iterator.next();
  assert.equal(next.value?.status, "awaiting_approval");
  await iterator.return?.(undefined);
});

test("the watcher stops once the run settles", async () => {
  reset();
  const done = run({ status: "completed", lastEventSeq: 3, report: "done" });
  ingest(done);
  useResearchRunStore.setState((state) => ({
    sessions: {
      ...state.sessions,
      [RUN_ID]: { ...state.sessions[RUN_ID], lastAppliedSeq: 3 },
    },
  }));

  const seen: string[] = [];
  for await (const value of watchResearchRun(RUN_ID)) {
    seen.push(value.status);
  }
  assert.deepEqual(seen, ["completed"]);
});

// A follower stopped by a permanent 4xx never restarts, so the run can never settle.
test("a follower that gave up surfaces instead of parking the watcher", async () => {
  reset();
  ingest(run({ status: "running" }));
  let threw: unknown = null;
  let exited = false;
  const consume = (async () => {
    try {
      for await (const snapshot of watchResearchRun(RUN_ID)) {
        void snapshot; // the chat adapter's loop
      }
    } catch (error) {
      threw = error;
    }
    exited = true;
  })();

  useResearchRunStore.getState().setConnectionError(RUN_ID, "Research run not found");
  await Promise.race([
    consume,
    new Promise((resolve) => setTimeout(resolve, 300)),
  ]);

  assert.equal(exited, true);
  assert.equal((threw as Error)?.message, "Research run not found");
});

// The park resolves immediately while dirty is set, so an absent session froze the tab.
test("watching a run the store does not have ends instead of spinning", async () => {
  reset();
  let timerFired = false;
  setTimeout(() => {
    timerFired = true;
  }, 20);

  for await (const snapshot of watchResearchRun("missing")) {
    assert.fail(`a run with no session must not yield ${snapshot.id}`);
  }
  await new Promise((resolve) => setTimeout(resolve, 60));

  assert.equal(timerFired, true, "the event loop must keep turning");
});

// A worker killed mid-call never writes phase.ended, so the resume has to close its row.
test("resuming a run closes a phase row orphaned by a crash", () => {
  reset();
  ingest(run());
  ingest(run(), event(1, "phase.started", { phase: "decision", callId: "a" }));
  const resumed = run({ status: "running", lastEventSeq: 2 });
  ingest(resumed, event(2, "run.started", { status: "running", resumed: true }, resumed));

  const [row] = activities().filter((a) => a.kind === "reasoning");
  assert.equal(row.state, "complete");
});

test("an aborted watcher ends without yielding again", async () => {
  reset();
  ingest(run());
  const controller = new AbortController();
  const iterator = watchResearchRun(RUN_ID, { signal: controller.signal })[
    Symbol.asyncIterator
  ]();
  await iterator.next();
  controller.abort();
  assert.equal((await iterator.next()).done, true);
});

test("a dropped phase.ended does not leave the previous row spinning", () => {
  // _note_phase writes phase.ended best-effort, so a new phase has to close the old row too.
  reset();
  ingest(run());
  ingest(
    run(),
    event(1, "phase.started", { phase: "planning", callId: "call-a" }),
  );
  ingest(
    run(),
    event(2, "phase.started", { phase: "decision", callId: "call-b" }),
  );

  const running = activities().filter((a) => a.state === "running");
  assert.equal(running.length, 1);
  assert.equal(running[0].title, "Choosing the next step");
  assert.equal(
    activities().find((a) => a.phase === "planning")?.state,
    "complete",
  );
});

test("a completed fetch step is not labelled as having found nothing", () => {
  // A fetch records an excerpt and never collects sources, so its sourceCount is always 0.
  reset();
  ingest(run());
  ingest(
    run(),
    event(1, "step.started", {
      stepPosition: 0,
      action: "fetch",
      title: "Reading a page",
      input: "https://example.com",
    }),
  );
  ingest(
    run(),
    event(2, "step.completed", {
      stepPosition: 0,
      action: "fetch",
      sourceCount: 0,
    }),
  );

  const step = activities().find((a) => a.kind === "step");
  assert.equal(step?.state, "complete");
  assert.equal(step?.detail, "Page read");
});
