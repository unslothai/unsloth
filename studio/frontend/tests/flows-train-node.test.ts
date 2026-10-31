// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { trainView, sparklinePath, adapterPathFor } = await import(
  "../src/features/flows/editor/train-status.ts"
);

const catalog = JSON.parse(
  readFileSync(new URL("./fixtures/flows-catalog.json", import.meta.url), "utf8"),
);
const finetune = catalog.nodes.find(
  (n: { type: string }) => n.type === "train.finetune",
);

const base = {
  status: "idle" as const,
  params: { base_model: "unsloth/Qwen3-0.6B" },
  hasDataset: true,
  outputs: null,
  partial: null,
  error: null,
};

test("Fine-tune leads with five fields and keeps the rest under Advanced", () => {
  const simple = finetune.params
    .filter((p: { advanced?: boolean }) => !p.advanced)
    .map((p: { id: string }) => p.id);
  assert.deepEqual(simple, [
    "base_model",
    "prompt_column",
    "response_column",
    "max_steps",
    "holdout_percent",
  ]);
  assert.ok(finetune.params.length > 15);
  assert.ok(
    finetune.outputs.some(
      (p: { id: string; type: string }) =>
        p.id === "holdout" && p.type === "dataset",
    ),
  );
});

test("the node walks needs dataset, setup, ready, training, adapter ready, stopped, failed", () => {
  assert.equal(trainView({ ...base, hasDataset: false }).state, "needs_dataset");
  assert.equal(trainView({ ...base, params: { base_model: "" } }).state, "setup");
  assert.equal(
    trainView({ ...base, params: { ...base.params, prompt_column: "q" } }).state,
    "setup",
  );
  assert.equal(
    trainView({
      ...base,
      params: { ...base.params, prompt_column: "q", response_column: "q" },
    }).detail,
    "Use two different columns",
  );
  assert.equal(
    trainView({
      ...base,
      params: { ...base.params, prompt_column: "q", custom_format_mapping: { q: "user" } },
    }).state,
    "ready",
  );
  assert.equal(trainView(base).state, "ready");
  const live = trainView({
    ...base,
    status: "running",
    partial: { metrics: { steps: 12, total: 60, loss: 1.23456, loss_history: [2, 1.5, 1.2] } },
  });
  assert.equal(live.state, "training");
  assert.equal(live.detail, "Step 12 of 60 · loss 1.235");
  assert.deepEqual(live.history, [2, 1.5, 1.2]);
  const done = trainView({
    ...base,
    status: "done",
    outputs: {
      checkpoint: { kind: "checkpoint", name: "run" },
      metrics: { loss: 0.9, train_rows: 270, holdout_rows: 30, loss_history: [2, 0.9] },
    },
  });
  assert.equal(done.state, "adapter_ready");
  assert.equal(done.detail, "loss 0.900 · 270 trained / 30 held out");
  assert.equal(trainView({ ...base, status: "cancelled" }).state, "stopped");
  assert.equal(
    trainView({ ...base, status: "failed", error: "OOM" }).detail,
    "OOM",
  );
});

test("sparkline spans the box and stays flat for a constant series", () => {
  assert.equal(sparklinePath([1], 100, 20), "");
  const path = sparklinePath([3, 2, 1], 100, 20);
  assert.ok(path.startsWith("M0.0,1.0") && path.endsWith("L100.0,19.0"), path);
  assert.ok(!sparklinePath([1, 1, 1], 100, 20).includes("NaN"));
});

test("Chat with it finds the adapter by the run folder's name", () => {
  const loras = [
    { adapter_path: "/out/run-1/checkpoint-10" },
    { adapter_path: "/out/run-1/" },
    { adapter_path: "C:\\out\\run-2" },
  ];
  assert.equal(adapterPathFor(loras, "run-1"), "/out/run-1/");
  assert.equal(adapterPathFor(loras, "run-2"), "C:\\out\\run-2");
  assert.equal(adapterPathFor(loras, "run-3"), null);
});
