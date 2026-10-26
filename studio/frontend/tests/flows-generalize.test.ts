// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { reduceEvents, useRuntimeStore } = await import(
  "../src/features/flows/stores/runtime-store.ts"
);
const { pickOptions } = await import(
  "../src/features/flows/editor/remote-options.ts"
);

const ev = (
  seq: number,
  type: string,
  nodeId: string | null,
  data: Record<string, unknown> = {},
) => ({ seq, run_id: "r1", type, node_id: nodeId, ts: seq, data }) as never;

function fresh() {
  useRuntimeStore.getState().clear();
  return useRuntimeStore.getState();
}

test("partial rows show while a step runs and give way to its output", () => {
  let state = {
    ...fresh(),
    ...reduceEvents(fresh(), [
      ev(1, "node.status", "b", { status: "running" }),
      ev(2, "node.partial", "b", {
        port: "table",
        value: [1],
        step: 1,
        total: 4,
      }),
      ev(3, "node.partial", "b", {
        port: "table",
        value: [1, 2],
        step: 2,
        total: 4,
      }),
    ]),
  };
  assert.deepEqual(state.nodes.b.partial, { table: [1, 2] });
  assert.deepEqual(state.nodes.b.partialStep, { step: 2, total: 4 });
  state = {
    ...state,
    ...reduceEvents(state, [
      ev(4, "node.output", "b", { outputs: { table: [1, 2, 3, 4] } }),
    ]),
  };
  assert.equal(state.nodes.b.partial, null);
  assert.deepEqual(state.nodes.b.outputs, { table: [1, 2, 3, 4] });
});

test("a reconnect restores a stopped step's last partial from the snapshot", () => {
  const state = reduceEvents(fresh(), [
    ev(9, "snapshot", null, {
      run: { status: "cancelled" },
      nodes: {
        b: {
          status: "cancelled",
          partial: {
            table: { port: "table", value: ["x"], step: 1, total: 3 },
          },
        },
      },
    }),
  ]);
  assert.deepEqual(state.nodes.b.partial, { table: ["x"] });
  assert.deepEqual(state.nodes.b.partialStep, { step: 1, total: 3 });
});

test("a new run clears the partial of the last one", () => {
  const state = reduceEvents(fresh(), [
    ev(1, "node.partial", "b", { port: "table", value: [1] }),
    ev(2, "node.status", "b", { status: "queued" }),
  ]);
  assert.equal(state.nodes.b.partial, null);
});

test("picker options come from a dotted path, mapping over lists", () => {
  const answer = {
    tasks: [{ id: "gsm8k" }, { id: "arc_easy" }, { name: "no id" }],
    groups: { mmlu: ["mmlu_stem", "mmlu_other"] },
  };
  assert.deepEqual(pickOptions(answer, "tasks[].id"), ["gsm8k", "arc_easy"]);
  assert.deepEqual(pickOptions(answer, "groups.mmlu"), [
    "mmlu_stem",
    "mmlu_other",
  ]);
  assert.deepEqual(pickOptions(["a", 2, null], ""), ["a", "2"]);
  assert.deepEqual(pickOptions({ tasks: "oops" }, "tasks[].id"), []);
});

test("a new run drops the last run's partial but keeps its outputs", () => {
  fresh();
  useRuntimeStore
    .getState()
    .applyEvents([
      ev(1, "node.status", "b", { status: "running" }),
      ev(2, "node.partial", "b", {
        port: "table",
        value: [1],
        step: 1,
        total: 4,
      }),
      ev(3, "node.status", "b", { status: "cancelled" }),
    ]);
  assert.ok(useRuntimeStore.getState().nodes.b.partial);
  useRuntimeStore.getState().startRun("r2");
  const node = useRuntimeStore.getState().nodes.b;
  assert.equal(node.partial, null);
  assert.equal(node.partialStep, null);
});
