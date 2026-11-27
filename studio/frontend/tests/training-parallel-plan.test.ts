// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The GPUs section: contract parsing, the one-line summary, the override checks, and the rule that a
 * 0 or 1 GPU host (or a backend without the multi-GPU routes) sends nothing new. */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  normalizeParallelCapabilities,
  normalizeParallelPlan,
  normalizeRunParallel,
  layoutExample,
  overrideIssue,
  parallelRequestFields,
  planSummary,
  runGpuCount,
  showsGpuSection,
  toggleGpuSelection,
} from "../src/features/training/lib/parallel-plan.ts";

const capabilitiesBody = (n: number) => ({
  platform: "linux",
  device: "cuda",
  distributed_backend: "nccl",
  multi_process: true,
  multi_process_reason: null,
  gpus: Array.from({ length: n }, (_, index) => ({
    index,
    name: "Tesla T4",
    total_gb: 14.6,
    free_gb: 14.3,
  })),
  identical_gpus: true,
  core: { ddp: true, fsdp2: true, tp: true, pp: true, cp: true, ep: false },
  modes: [
    { mode: "auto", available: true, reason: null, min_gpus: 1 },
    { mode: "ddp", available: true, reason: null, min_gpus: 2 },
    {
      mode: "ep",
      available: false,
      reason: "This Unsloth version has no expert parallel yet.",
      min_gpus: 2,
    },
  ],
});

const planBody = {
  requested_mode: "auto",
  mode: "ddp",
  world_size: 2,
  gpu_ids: [0, 1],
  spec: "dp=2",
  launcher: "multi_process",
  distributed_backend: "nccl",
  reason: "The model needs about 6.1 GB per GPU and each GPU has 14.3 GB free.",
  estimate: {
    required_gb: 6.1,
    per_gpu_gb: { single: 6.1, ddp: 6.1, fsdp2: 3.9 },
  },
  is_moe: false,
  warnings: [
    "The model needs about 6.1 GB per GPU, which may not fit; FSDP2 shards it.",
  ],
  notes: ["These GPUs differ; data parallel runs at the pace of the slowest."],
  modes: [
    {
      mode: "tp",
      available: false,
      reason: "tensor parallel with QLoRA ...",
      min_gpus: 2,
    },
  ],
  error: null,
};

const OFF = { enabled: false, mode: "auto" as const, gpuIds: null, spec: "" };

test("capabilities parse the contract body and drop unreadable GPUs", () => {
  const caps = normalizeParallelCapabilities({
    ...capabilitiesBody(2),
    gpus: [...capabilitiesBody(2).gpus, { index: -1 }, { name: "no index" }],
  });
  assert.ok(caps);
  assert.deepEqual(
    caps.gpus.map((g) => [g.index, g.name, g.totalGb, g.freeGb]),
    [
      [0, "Tesla T4", 14.6, 14.3],
      [1, "Tesla T4", 14.6, 14.3],
    ],
  );
  assert.equal(caps.multiProcess, true);
  const ep = caps.modes.find((m) => m.mode === "ep");
  assert.equal(ep?.available, false);
  assert.match(ep?.reason ?? "", /expert parallel/);
  assert.equal(normalizeParallelCapabilities({ detail: "Not Found" }), null);
  assert.equal(normalizeParallelCapabilities(null), null);
});

test("plan parses the resolved mode, the per-GPU estimate of that mode, and the error object", () => {
  const plan = normalizeParallelPlan(planBody);
  assert.ok(plan);
  assert.equal(plan.mode, "ddp");
  assert.equal(plan.worldSize, 2);
  assert.deepEqual(plan.gpuIds, [0, 1]);
  assert.equal(plan.perGpuGb, 6.1);
  assert.equal(plan.error, null);
  assert.deepEqual(plan.notes, [
    "These GPUs differ; data parallel runs at the pace of the slowest.",
  ]);
  assert.equal(plan.warnings.length, 1);
  const { notes: _omit, ...older } = planBody;
  assert.deepEqual(normalizeParallelPlan(older)?.notes, []);
  const refused = normalizeParallelPlan({
    ...planBody,
    mode: "tp",
    error: {
      code: "parallel_unsupported",
      mode: "tp",
      message: "Unsloth: tensor parallel is not supported with QLoRA.",
      alternative: "drop tp (DDP, or FSDP2)",
    },
  });
  assert.deepEqual(refused?.error, {
    code: "parallel_unsupported",
    message: "Unsloth: tensor parallel is not supported with QLoRA.",
    alternative: "drop tp (DDP, or FSDP2)",
  });
  // "auto" is a request value, never a resolved one.
  assert.equal(normalizeParallelPlan({ ...planBody, mode: "auto" }), null);
  assert.equal(normalizeParallelPlan({ ...planBody, mode: "zero3" }), null);
});

test("the summary names the mode, not a fit the planner may not have checked", () => {
  const plan = normalizeParallelPlan(planBody);
  assert.ok(plan);
  assert.deepEqual(planSummary(plan), {
    key: "studio.gpus.summary.ddp",
    params: { count: "2", spec: "dp=2" },
  });
  const en = readFileSync(
    new URL("../src/i18n/locales/en.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    en,
    /ddp: "Each GPU trains a full copy: data parallel on \{count\} GPUs"/,
  );
  // FSDP2 picked by hand for a small model must not claim the model is too big for one GPU.
  assert.doesNotMatch(en, /too big for one GPU/);
});

test("single and layer_split are one process, so the GPU count comes from gpu_ids", () => {
  assert.equal(
    runGpuCount({ mode: "layer_split", worldSize: 1, gpuIds: [0, 1, 2, 3] }),
    4,
  );
  assert.equal(runGpuCount({ mode: "single", worldSize: 1, gpuIds: [3] }), 1);
  assert.equal(
    runGpuCount({ mode: "fsdp2", worldSize: 4, gpuIds: [0, 1, 2, 3] }),
    4,
  );
});

test("one GPU or none shows no section and sends no new field", () => {
  assert.equal(showsGpuSection(0), false);
  assert.equal(showsGpuSection(1), false);
  assert.equal(showsGpuSection(2), true);
  const one = normalizeParallelCapabilities(capabilitiesBody(1));
  assert.deepEqual(parallelRequestFields(OFF, one), {});
  assert.deepEqual(
    parallelRequestFields(
      { enabled: true, mode: "ddp", gpuIds: [0], spec: "" },
      one,
    ),
    {},
  );
  // A backend without the routes never produced capabilities.
  assert.deepEqual(parallelRequestFields(OFF, null), {});
});

test("a multi-GPU host sends auto, so the run does what the section showed", () => {
  const two = normalizeParallelCapabilities(capabilitiesBody(2));
  assert.deepEqual(parallelRequestFields(OFF, two), { parallel_mode: "auto" });
});

test("the override sends its mode, the spec only for custom, and only GPUs that still exist", () => {
  const four = normalizeParallelCapabilities(capabilitiesBody(4));
  assert.deepEqual(
    parallelRequestFields(
      { enabled: true, mode: "fsdp2", gpuIds: null, spec: "dp=2" },
      four,
    ),
    { parallel_mode: "fsdp2" },
  );
  assert.deepEqual(
    parallelRequestFields(
      { enabled: true, mode: "custom", gpuIds: [0, 1, 7], spec: " dp=2,tp=2 " },
      four,
    ),
    { parallel_mode: "custom", parallel_spec: "dp=2,tp=2", gpu_ids: [0, 1] },
  );
});

test("browser-side override checks mirror parallel_gpu_ids", () => {
  const all = [0, 1, 2, 3];
  assert.equal(overrideIssue(OFF, all), null);
  assert.equal(
    overrideIssue(
      { enabled: true, mode: "single", gpuIds: null, spec: "" },
      all,
    ),
    null, // every GPU ticked: the backend trains on the freest one
  );
  assert.equal(
    overrideIssue(
      { enabled: true, mode: "single", gpuIds: [1, 2], spec: "" },
      all,
    ),
    "studio.gpus.error.singleNeedsOne",
  );
  assert.equal(
    overrideIssue(
      { enabled: true, mode: "single", gpuIds: [2], spec: "" },
      all,
    ),
    null,
  );
  assert.equal(
    overrideIssue({ enabled: true, mode: "ddp", gpuIds: [1], spec: "" }, all),
    "studio.gpus.error.needsTwo",
  );
  assert.equal(
    overrideIssue({ enabled: true, mode: "auto", gpuIds: [], spec: "" }, all),
    "studio.gpus.error.noGpu",
  );
  assert.equal(
    overrideIssue(
      { enabled: true, mode: "custom", gpuIds: null, spec: "  " },
      all,
    ),
    "studio.gpus.error.customSpec",
  );
});

test("the layout example is one every run accepts (data parallel over every GPU)", () => {
  assert.equal(layoutExample(2), "dp=2");
  assert.equal(layoutExample(3), "dp=3");
  assert.equal(layoutExample(4), "dp=4");
  assert.equal(layoutExample(8), "dp=8");
});

test("the plan request carries packing, which decides pipeline parallel", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/training/hooks/use-parallel-plan.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const fields = hook.slice(
    hook.indexOf("const PLAN_FIELDS"),
    hook.indexOf("] as const"),
  );
  assert.match(fields, /"packing"/);
  const key = hook.slice(
    hook.indexOf("const planKey"),
    hook.indexOf("const override"),
  );
  assert.match(key, /s\.packing/);
});

test("ticking every GPU again returns to 'all GPUs' (null)", () => {
  const all = [0, 1, 2, 3];
  const without2 = toggleGpuSelection(null, 2, all);
  assert.deepEqual(without2, [0, 1, 3]);
  assert.equal(toggleGpuSelection(without2, 2, all), null);
});

test("run status parallel: null for legacy runs, parsed otherwise", () => {
  assert.equal(normalizeRunParallel(null), null);
  assert.equal(normalizeRunParallel(undefined), null);
  assert.deepEqual(
    normalizeRunParallel({
      mode: "ddp",
      world_size: 2,
      gpu_ids: [0, 1],
      spec: "dp=2",
      launcher: "multi_process",
    }),
    { mode: "ddp", worldSize: 2, gpuIds: [0, 1], spec: "dp=2" },
  );
});

test("the wizard section renders nothing until capabilities list two or more GPUs", () => {
  const src = readFileSync(
    new URL(
      "../src/features/studio/sections/gpus-section.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const body = src.slice(src.indexOf("export function GpusSection"));
  assert.match(body, /!showsGpuSection\(systemGpuCount\)/);
  assert.match(body, /!showsGpuSection\(capabilities\.gpus\.length\)/);
  assert.match(body, /return null;/);
});

test("a GPU override edit while Start is pending counts as an edit, like any other form change", () => {
  const src = readFileSync(
    new URL(
      "../src/features/training/lib/start-fresh-training-run.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const capture = src.slice(
    src.indexOf("function captureTrainingStartInputs"),
    src.indexOf("type TrainingStartInputs"),
  );
  assert.match(
    capture,
    /parallelOverride: useTrainingParallelStore\.getState\(\)\.override/,
  );
  assert.doesNotMatch(capture, /currentParallelRequestFields|\.capabilities/);
});

test("an auto layer split with gpu_ids null counts the host's GPUs, not 1", () => {
  // Real backend body: auto on 2 x T4 for a 40 GB model.
  const plan = normalizeParallelPlan({
    requested_mode: "auto",
    mode: "layer_split",
    world_size: 1,
    gpu_ids: null,
    spec: null,
    error: null,
  });
  assert.ok(plan);
  assert.equal(runGpuCount(plan, 2), 2);
  assert.equal(planSummary(plan, String, 2).params.count, "2");
});

test("a refused choice shows only the refusal, not the mode it resolved to", () => {
  const src = readFileSync(
    new URL(
      "../src/features/studio/sections/gpus-section.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const line = src.slice(
    src.indexOf("function PlanLine"),
    src.indexOf("function ModeSelect"),
  );
  assert.match(
    line,
    /\{!plan\.error && \(\s*<div className="flex flex-wrap items-center gap-2">/,
  );
});
