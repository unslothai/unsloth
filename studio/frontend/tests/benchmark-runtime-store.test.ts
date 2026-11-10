// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./benchmark-runtime-resolver.mjs", import.meta.url);

const { calls } = await import("./helpers/store-stubs/benchmark-api.ts");
const { useBenchmarkRuntimeStore } = await import(
  "../src/features/benchmark/stores/benchmark-runtime-store.ts"
);

test("after its own run ends, the tab follows a run started elsewhere", async () => {
  const store = useBenchmarkRuntimeStore;
  await store.getState().run("org/model", "hf", "arc_easy");
  assert.equal(store.getState().phase, "success");
  assert.equal(store.getState().ownsRun, false);

  store.getState().applyBackendStatus({ is_benchmark_active: true, last_op_seq: 2 });
  const attached = store.getState();
  assert.equal(attached.phase, "running");
  assert.equal(attached.isRunning, true);
  assert.equal(attached.task, null);
  assert.equal(attached.result, null);

  await store.getState().requestCancel();
  assert.deepEqual(calls, ["run", "cancel"]);
  store
    .getState()
    .applyBackendStatus({ is_benchmark_active: false, last_op_seq: 2, last_op_status: "cancelled" });
  assert.equal(store.getState().phase, "canceled");
  assert.equal(store.getState().isRunning, false);
});
