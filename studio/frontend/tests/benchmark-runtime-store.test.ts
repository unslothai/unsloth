// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test, { mock } from "node:test";

register("./benchmark-runtime-resolver.mjs", import.meta.url);

const {
  calls,
  runParams,
  setRunDrops,
  setStatusGate,
  setStatusScript,
  statusTokens,
} = await import(
  "./helpers/store-stubs/benchmark-api.ts"
);
const { useBenchmarkRuntimeStore } = await import(
  "../src/features/benchmark/stores/benchmark-runtime-store.ts"
);

test("after its own run ends, the tab follows a run started elsewhere", async () => {
  const store = useBenchmarkRuntimeStore;
  await store.getState().run("org/model", "hf", "arc_easy");
  assert.equal(store.getState().phase, "success");
  assert.equal(store.getState().ownsRun, false);

  store
    .getState()
    .applyBackendStatus({ is_benchmark_active: true, last_op_seq: 2 });
  const attached = store.getState();
  assert.equal(attached.phase, "running");
  assert.equal(attached.isRunning, true);
  assert.equal(attached.task, null);
  assert.equal(attached.result, null);

  await store.getState().requestCancel();
  assert.deepEqual(calls, ["run", "cancel"]);
  store
    .getState()
    .applyBackendStatus({
      is_benchmark_active: false,
      last_op_seq: 2,
      last_op_status: "cancelled",
    });
  assert.equal(store.getState().phase, "canceled");
  assert.equal(store.getState().isRunning, false);
});

test("a picked quant is pinned as expected_model so a swapped quant fails the run", async () => {
  const { evalsExpectedModel } = await import(
    "../src/features/benchmark/lib/model-labels.ts"
  );
  const expected = evalsExpectedModel("unsloth/Qwen3-0.6B-GGUF", "UD-Q4_K_XL");
  assert.equal(expected, "unsloth/Qwen3-0.6B-GGUF:UD-Q4_K_XL");
  assert.equal(evalsExpectedModel("unsloth/Qwen3-0.6B", null), null);

  const store = useBenchmarkRuntimeStore;
  store.getState().reset();
  await store.getState().run("unsloth/Qwen3-0.6B-GGUF", "local", "arc_easy", {
    expected_model: expected,
  });
  assert.equal(
    runParams.at(-1)?.expected_model,
    "unsloth/Qwen3-0.6B-GGUF:UD-Q4_K_XL",
  );
  assert.equal(runParams.at(-1)?.checkpoint_path, "unsloth/Qwen3-0.6B-GGUF");

  await store
    .getState()
    .run("unsloth/Qwen3-0.6B", "local", "arc_easy", { expected_model: null });
  assert.equal("expected_model" in (runParams.at(-1) ?? {}), false);
});

test("Cancel while the pre-run status probe is pending never posts the evaluation", async () => {
  const store = useBenchmarkRuntimeStore;
  store.getState().reset();
  calls.length = 0;
  let release!: () => void;
  setStatusGate(new Promise<void>((r) => (release = r)));
  const running = store.getState().run("org/model", "hf", "arc_easy");
  // Let run() reach the pending status probe.
  while (store.getState().phase !== "starting")
    await new Promise((r) => setTimeout(r, 0));
  await store.getState().requestCancel();
  setStatusGate(null);
  release();
  await running;
  assert.deepEqual(calls, ["cancel"]);
  assert.equal(store.getState().phase, "canceled");
  assert.equal(store.getState().isRunning, false);
});

test("a dropped evaluation request does not adopt another tab's run as its own", async () => {
  const store = useBenchmarkRuntimeStore;
  store.getState().reset();
  mock.timers.enable({ apis: ["setTimeout", "Date"] });
  try {
    setRunDrops(true);
    // Baseline probe, then another tab's run is active and finishes successfully.
    setStatusScript([
      { is_benchmark_active: false, last_op_seq: 0 },
      { is_benchmark_active: true, last_op_seq: 1, run_token_matches: false },
      {
        is_benchmark_active: false,
        last_op_seq: 1,
        last_op_status: "success",
        run_token_matches: false,
      },
    ]);
    let done = false;
    const running = store
      .getState()
      .run("org/model", "hf", "arc_easy")
      .finally(() => (done = true));
    for (let i = 0; i < 200 && !done; i++) {
      await new Promise((r) => setImmediate(r));
      mock.timers.tick(1500);
    }
    await running;
    assert.notEqual(store.getState().result, "Benchmark completed");
    assert.equal(store.getState().phase, "error");
    const token = runParams.at(-1)?.run_token;
    assert.equal(typeof token, "string");
    assert.equal(statusTokens.at(-1), token);
  } finally {
    mock.timers.reset();
    setRunDrops(false);
    setStatusScript([]);
  }
});
