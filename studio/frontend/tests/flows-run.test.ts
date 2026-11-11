// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { mock } from "node:test";

import { doc, node } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";
import { loadWithStubs, stubEveryImport } from "./helpers/module-stubs.ts";

registerBundlerResolver();

const docStore = await import("../src/features/flows/stores/doc-store.ts");
const runtimeStore = await import("../src/features/flows/stores/runtime-store.ts");
const uiStore = await import("../src/features/flows/stores/ui-store.ts");
const { useDocStore } = docStore;
const { useRuntimeStore, progressFraction } = runtimeStore;
const { useUiStore } = uiStore;

type Plan = { models: Record<string, unknown>[]; errors: unknown[] };

/** run-controller with a fake backend: start requests stay pending until the test settles them. */
async function runController(plan: () => Promise<Plan>) {
  const starts: { resolve: (v: { run_id: string }) => void }[] = [];
  const notices: string[] = [];
  const runs = loadWithStubs<{
    runFlow: (o?: Record<string, unknown>) => Promise<void>;
    detachRun: () => void;
  }>(
    new URL("../src/features/flows/editor/run-controller.ts", import.meta.url),
    stubEveryImport({
      sonner: {
        toast: {
          message: (m: string) => notices.push(m),
          error: (m: string) => notices.push(m),
        },
      },
      "../api": {
        planFlow: plan,
        startRun: () => new Promise((resolve) => starts.push({ resolve })),
        followRunEvents: () => new Promise(() => undefined),
      },
      "../stores/doc-store": docStore,
      "../stores/runtime-store": runtimeStore,
      "../stores/ui-store": uiStore,
      "../ux-rules": await import("../src/features/flows/ux-rules.ts"),
      "../stores/pins-store": { pinsForRun: () => ({}) },
    }),
  );
  useDocStore.getState().load(doc([node("t", "text.template")]));
  useRuntimeStore.setState({ runStatus: "idle" } as never);
  useUiStore.getState().setPendingRun(null);
  return { runs, starts, notices };
}

test("a second Run click while the first start request is pending is ignored", async () => {
  const { runs, starts, notices } = await runController(async () => ({ models: [], errors: [] }));
  const first = runs.runFlow({ downloadsConfirmed: true });
  await runs.runFlow({ downloadsConfirmed: true });
  assert.equal(starts.length, 1, "the second click starts nothing");
  assert.deepEqual(notices, ["A run is already in progress"]);

  starts[0].resolve({ run_id: "r1" });
  await first;
  useRuntimeStore.setState({ runStatus: "idle" } as never);
  void runs.runFlow({ downloadsConfirmed: true });
  await Promise.resolve();
  assert.equal(starts.length, 2, "a click after the start settles runs again");
});

test("a start that settles after the editor switched flows is not shown in the new flow", async () => {
  const { runs, starts } = await runController(async () => ({ models: [], errors: [] }));
  useRuntimeStore.setState({ runId: null } as never);
  const first = runs.runFlow({ downloadsConfirmed: true });
  await Promise.resolve();
  assert.equal(starts.length, 1);
  runs.detachRun();
  useDocStore.getState().load({ ...doc([node("t", "text.template")]), id: "f2" });
  starts[0].resolve({ run_id: "r_of_f1" });
  await first;
  assert.equal(useRuntimeStore.getState().runId, null, "flow f1's run stays out of flow f2's editor");
  assert.equal(useRuntimeStore.getState().runStatus, "idle");
});

test("Run in the next flow is not blocked by the last flow's pending start", async () => {
  const { runs, starts, notices } = await runController(async () => ({ models: [], errors: [] }));
  const first = runs.runFlow({ downloadsConfirmed: true });
  await Promise.resolve();
  runs.detachRun();
  useDocStore.getState().load({ ...doc([node("t", "text.template")]), id: "f2" });
  const second = runs.runFlow({ downloadsConfirmed: true });
  await Promise.resolve();
  assert.equal(starts.length, 2, "flow f2 starts its own run");
  assert.deepEqual(notices, []);
  starts[0].resolve({ run_id: "r_of_f1" });
  await first;
  void runs.runFlow({ downloadsConfirmed: true });
  await Promise.resolve();
  assert.equal(starts.length, 2, "f1's start settling does not unlock f2's pending one");
  starts[1].resolve({ run_id: "r_of_f2" });
  await second;
  assert.equal(useRuntimeStore.getState().runId, "r_of_f2");
});

test("Run asks the plan first and holds the run while models must download", async () => {
  const missing = { kind: "image", repo: "b/image", downloaded: false, loaded: false };
  let plans = 0;
  const { runs, starts } = await runController(async () => {
    plans += 1;
    return { models: [missing], errors: [] };
  });
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    await runs.runFlow();
    assert.equal(plans, 1);
    assert.equal(starts.length, 0, "nothing starts while a model is missing");
    const pending = useUiStore.getState().pendingRun;
    assert.ok(pending, "the download dialog is asked for");
    assert.deepEqual(pending.plan.models, [missing]);

    void runs.runFlow({ ...pending.options, downloadsConfirmed: true });
    await Promise.resolve();
    assert.equal(starts.length, 1, "a confirmed run starts without asking again");
    assert.equal(plans, 1);
  } finally {
    mock.timers.reset();
  }
});

test("a clear plan is remembered, so the next Run skips the check", async () => {
  let plans = 0;
  const { runs, starts } = await runController(async () => {
    plans += 1;
    return { models: [], errors: [] };
  });
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const first = runs.runFlow();
    for (let i = 0; i < 5 && !starts.length; i++) await Promise.resolve();
    starts[0].resolve({ run_id: "r1" });
    await first;
    useRuntimeStore.setState({ runStatus: "idle" } as never);
    void runs.runFlow();
    await Promise.resolve();
    assert.equal(plans, 1);
    assert.equal(starts.length, 2);
  } finally {
    mock.timers.reset();
  }
});

test("a run snapshot's progress object becomes a fraction", () => {
  assert.equal(progressFraction({ fraction: 0.25, phase: "x" }), 0.25);
  assert.equal(progressFraction({ step: 3, total: 4 }), 0.75);
  assert.equal(progressFraction({ phase: "loading" }), null);
  assert.equal(progressFraction(null), null);
  assert.equal(progressFraction(0.5), 0.5);
});
