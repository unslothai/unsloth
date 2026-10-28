// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type { ParamSpec } from "../src/features/flows/types.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { trainView, sparklinePath, rewardMax, rewardText, adapterPathFor } =
  await import(
  "../src/features/flows/editor/train-status.ts"
);
const { paramVisible } = await import(
  "../src/features/flows/editor/param-visible.ts"
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

const simpleFor = (objective: string) =>
  finetune.params
    .filter(
      (p: ParamSpec) =>
        !p.advanced && paramVisible(p, { objective }, finetune.params),
    )
    .map((p: { id: string }) => p.id);

test("Fine-tune leads with a few fields per objective and keeps the rest under Advanced", () => {
  assert.deepEqual(simpleFor("sft"), [
    "base_model",
    "objective",
    "prompt_column",
    "response_column",
    "max_steps",
    "holdout_percent",
  ]);
  assert.deepEqual(simpleFor("grpo"), [
    "base_model",
    "objective",
    "prompt_column",
    "answer_column",
    "rewards",
    "max_steps",
    "holdout_percent",
  ]);
  assert.deepEqual(simpleFor("dpo"), simpleFor("orpo"));
  assert.ok(simpleFor("dpo").includes("chosen_column"));
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

test("GRPO shows the reward against its best, with where it started", () => {
  const grpo = { ...base, params: { ...base.params, objective: "grpo", rewards: [] } };
  assert.equal(trainView(grpo).detail, "Pick a reward");
  const older = { ...base, params: { ...base.params, objective: "grpo" } };
  assert.equal(trainView(older).state, "ready");
  const live = trainView({
    ...base,
    status: "running",
    partial: {
      metrics: { steps: 4, total: 40, loss: 0.01, reward: 1, reward_max: 2, reward_history: [0.2, 1] },
    },
  });
  assert.equal(live.detail, "Step 4 of 40 · reward 1.00 / 2");
  assert.ok(live.reward);
  assert.deepEqual(live.history, [0.2, 1]);
  const done = trainView({
    ...base,
    status: "done",
    outputs: {
      checkpoint: { kind: "checkpoint", name: "run" },
      metrics: { reward: 1.8, reward_start: 0.4, reward_max: 2.5, reward_history: [0.4, 1.8] },
    },
  });
  assert.equal(done.detail, "reward 1.80 / 2.5 · from 0.40");
  assert.equal(rewardText(0.5, null), "reward 0.50");
});

test("reward max is the best weighted sum of the rules", () => {
  const rules = {
    "exact-answer": { type: "exact_match", score: { match: 2, miss: 0 }, missing: 0 },
    "numeric-close": { type: "numeric", bands: [{ within: 0, score: 3 }], else: -1, missing: -2 },
  };
  assert.equal(rewardMax([{ name: "exact-answer", weight: 1 }], rules), 2);
  assert.equal(rewardMax([{ name: "numeric-close", weight: -1 }], rules), 2);
  assert.equal(rewardMax([{ name: "gone", weight: 1 }], rules), null);
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
