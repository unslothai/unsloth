// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./recipe-tracker-resolver.mjs", import.meta.url);
(globalThis as { window?: unknown }).window ??= globalThis;

const { deps } = await import("./helpers/recipe-tracker-deps.ts");
const { trackRecipeExecution } = await import(
  "../src/features/recipe-studio/executions/tracker.ts"
);
const { createBaseExecutionRecord } = await import(
  "../src/features/recipe-studio/executions/runtime.ts"
);

type Terminal = {
  type: "job.completed" | "job.error" | "job.cancelled";
  payload?: Record<string, unknown>;
};

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

async function run(options: {
  kind: "preview" | "full";
  terminal: Terminal;
  terminalStatus: string;
  at: { call: number; when: "start" | "resolved"; ms: number };
  statusLatencyMs?: number;
}) {
  const calls: string[] = [];
  deps.calls = calls;
  let statusCalls = 0;
  let emitted = false;
  let emittedAt = 0;
  let firstCallAfterEmit = 0;
  let emit: () => void = () => undefined;
  const schedule = () => setTimeout(() => emit(), options.at.ms);

  deps.getRecipeJobStatus = (async (jobId: string) => {
    const index = statusCalls++;
    if (emitted && !firstCallAfterEmit) firstCallAfterEmit = performance.now();
    calls.push(`status#${index}`);
    if (options.at.call === index && options.at.when === "start") schedule();
    if (index === options.at.call && options.statusLatencyMs) {
      await sleep(options.statusLatencyMs);
    }
    const status = emitted ? options.terminalStatus : "active";
    calls.push(`status#${index}->${status}`);
    if (options.at.call === index && options.at.when === "resolved") schedule();
    return { job_id: jobId, status };
  }) as never;
  deps.getRecipeJobDataset = (async (
    _jobId: string,
    page: { limit: number; offset: number },
  ) => {
    calls.push(`dataset(${page.limit},${page.offset})`);
    return { dataset: [{ a: 1 }], total: 1 };
  }) as never;
  deps.getRecipeJobAnalysis = (async () => {
    calls.push("analysis");
    return { num_records: 1 };
  }) as never;
  deps.streamRecipeJobEvents = (({
    signal,
    onEvent,
  }: {
    signal: AbortSignal;
    onEvent: (event: unknown) => void;
  }) =>
    new Promise<void>((resolve) => {
      signal.addEventListener("abort", () => resolve());
      emit = () => {
        if (emitted || signal.aborted) return;
        emitted = true;
        emittedAt = performance.now();
        calls.push(`sse:${options.terminal.type}`);
        onEvent({
          event: "message",
          id: 1,
          payload: { type: options.terminal.type, ...options.terminal.payload },
        });
      };
    })) as never;

  const result = await trackRecipeExecution({
    label: "Preview",
    kind: options.kind,
    rows: 1,
    jobId: "job-1",
    initialExecution: createBaseExecutionRecord({
      recipeId: "recipe-1",
      kind: options.kind,
      rows: 1,
      currentSignature: "sig",
    }),
    notify: true,
    onUpsert: (record: { status: string }) =>
      calls.push(`upsert:${record.status}`),
    onSetPreviewErrors: () => calls.push("previewErrors"),
    onPreviewSuccess: () => calls.push("previewSuccess"),
  });
  const returnedAt = performance.now();
  const pendingTimers = process
    .getActiveResourcesInfo()
    .filter((kind) => kind === "Timeout").length;
  return {
    calls,
    result,
    pendingTimers,
    sseToNextStatusMs: firstCallAfterEmit
      ? firstCallAfterEmit - emittedAt
      : null,
    sseToReturnMs: returnedAt - emittedAt,
  };
}

const COMPLETED_TAIL = [
  "upsert:completed",
  "status#1",
  "status#1->completed",
  "status#2",
  "status#2->completed",
  "status#3",
  "status#3->completed",
  "analysis",
  "dataset(20,0)",
  "upsert:completed",
  "previewErrors",
  "previewSuccess",
  "toastSuccess:Preview generated (1 rows).",
];

test("job.completed during the poll sleep leaves the loop at once and keeps the final reads", async () => {
  const run1 = await run({
    kind: "preview",
    terminal: { type: "job.completed" },
    terminalStatus: "completed",
    at: { call: 0, when: "resolved", ms: 100 },
  });
  assert.deepEqual(run1.result, { success: true, terminal: true });
  assert.deepEqual(run1.calls, [
    "status#0",
    "status#0->active",
    "upsert:active",
    "dataset(20,0)",
    "upsert:active",
    "sse:job.completed",
    ...COMPLETED_TAIL,
  ]);
  assert.ok(run1.sseToNextStatusMs !== null && run1.sseToNextStatusMs < 300);
  assert.equal(run1.pendingTimers, 0);
});

for (const [type, status, toast] of [
  ["job.error", "error", "toastError:Preview failed|boom"],
  [
    "job.cancelled",
    "cancelled",
    "toastError:Preview cancelled|The execution was cancelled.",
  ],
] as const) {
  test(`${type} during the poll sleep returns at once with no further reads`, async () => {
    const outcome = await run({
      kind: "preview",
      terminal: {
        type,
        payload: type === "job.error" ? { error: "boom" } : {},
      },
      terminalStatus: status,
      at: { call: 0, when: "resolved", ms: 100 },
    });
    assert.deepEqual(outcome.result, { success: false, terminal: true });
    assert.deepEqual(outcome.calls, [
      "status#0",
      "status#0->active",
      "upsert:active",
      "dataset(20,0)",
      "upsert:active",
      `sse:${type}`,
      `upsert:${status}`,
      `upsert:${status}`,
      toast,
    ]);
    assert.ok(outcome.sseToReturnMs < 300);
    assert.equal(outcome.pendingTimers, 0);
  });
}

test("a terminal event during an in-flight status read still breaks after that read", async () => {
  const outcome = await run({
    kind: "preview",
    terminal: { type: "job.completed" },
    terminalStatus: "completed",
    at: { call: 0, when: "start", ms: 50 },
    statusLatencyMs: 200,
  });
  assert.deepEqual(outcome.result, { success: true, terminal: true });
  assert.deepEqual(outcome.calls, [
    "status#0",
    "sse:job.completed",
    "upsert:completed",
    "status#0->completed",
    ...COMPLETED_TAIL.slice(1),
  ]);
});
