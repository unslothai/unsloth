// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ParamSpec } from "../src/features/flows/types.ts";
import { specs } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { trainView, sparklinePath, adapterPathFor, chatHandoffFor } = await import(
  "../src/features/flows/editor/train-status.ts"
);
const { paramVisible } = await import("../src/features/flows/editor/param-visible.ts");

const finetune = specs["train.finetune"];
const params: ParamSpec[] = finetune.params;

const base = {
  status: "idle" as const,
  params: { base_model: "unsloth/Qwen3-0.6B" },
  hasDataset: true,
  outputs: null,
  partial: null,
  error: null,
};
type TrainInput = Parameters<typeof trainView>[0];
const view = (over: Partial<TrainInput>) => trainView({ ...base, ...over } as TrainInput);
const withParams = (extra: Record<string, unknown>) => view({ params: { ...base.params, ...extra } });

const visible = (values: Record<string, unknown>, simpleOnly = false) =>
  params.filter((p) => !(simpleOnly && p.advanced) && paramVisible(p, values, params)).map((p) => p.id);

test("Fine-tune leads with a few fields and keeps the rest under Advanced", () => {
  assert.deepEqual(visible({}, true), [
    "base_model",
    "prompt_column",
    "response_column",
    "max_steps",
    "holdout_percent",
  ]);
  assert.ok(params.length > 15);
  assert.ok(finetune.outputs.some((p: { id: string; type: string }) => p.id === "holdout" && p.type === "dataset"));
});

test("the node walks needs dataset, setup, ready, training, adapter ready, stopped, failed", () => {
  assert.equal(view({ hasDataset: false }).state, "needs_dataset");
  assert.equal(view({ params: { base_model: "" } }).state, "setup");
  assert.equal(withParams({ prompt_column: "q" }).state, "setup");
  assert.equal(withParams({ prompt_column: "q", response_column: "q" }).detail, "Use two different columns");
  assert.equal(withParams({ prompt_column: "q", custom_format_mapping: { q: "user" } }).state, "ready");
  assert.equal(trainView(base).state, "ready");
  const live = view({
    status: "running",
    partial: { metrics: { steps: 12, total: 60, loss: 1.23456, loss_history: [2, 1.5, 1.2] } },
  });
  assert.equal(live.state, "training");
  assert.equal(live.detail, "Step 12 of 60 · loss 1.235");
  assert.deepEqual(live.history, [2, 1.5, 1.2]);
  const done = view({
    status: "done",
    outputs: {
      checkpoint: { kind: "checkpoint", name: "run" },
      metrics: { loss: 0.9, train_rows: 270, holdout_rows: 30, loss_history: [2, 0.9] },
    },
  });
  assert.equal(done.state, "adapter_ready");
  assert.equal(done.detail, "loss 0.900 · 270 trained / 30 held out");
  assert.equal(view({ status: "cancelled" }).state, "stopped");
  assert.equal(view({ status: "failed", error: "OOM" }).detail, "OOM");
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

test("Chat with it hands over nothing when the run cannot be found, so Chat never picks another adapter", () => {
  const checkpoint = { base_model: "unsloth/Qwen3-0.6B", name: "run-1" };
  assert.deepEqual(chatHandoffFor({ loras: [{ adapter_path: "/out/run-1" }] }, checkpoint), {
    baseModel: "unsloth/Qwen3-0.6B",
    outputDir: "/out/run-1",
  });
  // The adapter scan failed, or does not list the run: no handoff with a null folder.
  assert.equal(chatHandoffFor(null, checkpoint), null);
  assert.equal(chatHandoffFor({ loras: [{ adapter_path: "/out/run-2" }] }, checkpoint), null);
  assert.equal(chatHandoffFor({ loras: [] }, undefined), null);
});

test("Advanced holds every Train page setting, grouped in the Train page's order", () => {
  const groups: string[] = [];
  for (const p of params) {
    if (p.advanced) assert.ok(p.group, `${p.id} has no group`);
    if (p.group && groups.at(-1) !== p.group) {
      assert.ok(!groups.includes(p.group), `${p.group} is split`);
      groups.push(p.group);
    }
  }
  assert.deepEqual(groups, [
    "Model", "Dataset", "Training", "LoRA Settings", "Optimization", "Schedule", "Memory",
    "Speculative Decoding Drafter",
  ]);
  const ids = new Set(params.map((p) => p.id));
  for (const id of [
    "project_name", "dataset_slice_start", "eval_split", "embedding_learning_rate",
    "lora_dropout", "lora_variant", "target_modules", "finetune_vision_layers",
    "optim", "lr_scheduler_type", "save_steps", "eval_steps", "early_stopping_patience",
    "random_seed", "packing", "gradient_checkpointing", "offload_layers", "offload_vram_gb",
    "prefetch_depth", "drafter_enabled", "drafter_draft_steps",
  ])
    assert.ok(ids.has(id), id);
});

const visibility: [string, Record<string, unknown>, string, boolean][] = [
  ["drafter epochs hidden by default", {}, "drafter_epochs", false],
  ["drafter epochs shown with the drafter on", { drafter_enabled: true }, "drafter_epochs", true],
  ["LoRA rank hidden for full fine-tuning", { training_type: "full" }, "lora_r", false],
  ["embedding LR shown for CPT", { training_type: "cpt" }, "embedding_learning_rate", true],
  ["vision layers shown for vision", { task: "vision" }, "finetune_vision_layers", true],
  ["target modules hidden for vision", { task: "vision" }, "target_modules", false],
  ["offload VRAM shown for auto offload", { offload_layers: "auto" }, "offload_vram_gb", true],
  ["offload VRAM hidden for count offload", { offload_layers: "count" }, "offload_vram_gb", false],
];
for (const [name, values, id, shown] of visibility) {
  test(`drafter fields show only for SFT with the drafter on, LoRA fields only for adapter methods: ${name}`, () => {
    assert.equal(visible(values).includes(id), shown);
  });
}

test("a reused adapter says it was not retrained", () => {
  const outputs = { checkpoint: { kind: "checkpoint", name: "run" }, metrics: { loss: 1.2, steps: 60, total: 60 } };
  const cached = view({ status: "cached", outputs });
  assert.equal(cached.state, "adapter_ready");
  assert.match(String(cached.detail), /^Reused, not retrained/);
  assert.doesNotMatch(String(view({ status: "done", outputs }).detail), /Reused/);
});
