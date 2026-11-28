// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { DEFAULT_HYPERPARAMS } from "../src/config/training.ts";
import {
  MAX_SUGGESTED_STEPS,
  MIN_SUGGESTED_STEPS,
  llmStepGuidance,
  planDataParallelSize,
  runDataParallelSize,
  trainedRowCount,
} from "../src/features/training/lib/step-guidance.ts";
import { readSrc } from "./helpers/kit.ts";

test("one pass is ceil(rows / (batch x grad accum)) steps", () => {
  // the default 4 x 8 = 32 rows per step
  assert.deepEqual(llmStepGuidance(10_000, 4, 8), {
    rows: 10_000,
    effectiveBatch: 32,
    passSteps: 313,
    suggestedSteps: 313,
    kind: "pass",
  });
  assert.equal(llmStepGuidance(3200, 4, 8)?.suggestedSteps, 100);
  assert.equal(llmStepGuidance(3201, 4, 8)?.suggestedSteps, 101);
  assert.equal(llmStepGuidance(500, 1, 1)?.suggestedSteps, 500);
});

test("a tiny set gets the floor, a huge one the cap, and the pass count stays exact", () => {
  const tiny = llmStepGuidance(100, 4, 8);
  assert.equal(tiny?.kind, "floor");
  assert.equal(tiny?.passSteps, 4);
  assert.equal(tiny?.suggestedSteps, MIN_SUGGESTED_STEPS);
  const huge = llmStepGuidance(1_000_000, 4, 8);
  assert.equal(huge?.kind, "cap");
  assert.equal(huge?.passSteps, 31_250);
  assert.equal(huge?.suggestedSteps, MAX_SUGGESTED_STEPS);
  assert.equal(llmStepGuidance(MIN_SUGGESTED_STEPS * 32, 4, 8)?.kind, "pass");
  assert.equal(llmStepGuidance(MAX_SUGGESTED_STEPS * 32, 4, 8)?.kind, "pass");
});

test("unknown rows give no guidance and bad batch values count as 1", () => {
  assert.equal(llmStepGuidance(null, 4, 8), null);
  assert.equal(llmStepGuidance(undefined, 4, 8), null);
  assert.equal(llmStepGuidance(0, 4, 8), null);
  assert.equal(llmStepGuidance(Number.NaN, 4, 8), null);
  assert.equal(llmStepGuidance(64, 0, Number.NaN)?.effectiveBatch, 1);
});

test("the trained rows follow the inclusive train slice", () => {
  assert.equal(trainedRowCount(1000), 1000);
  assert.equal(trainedRowCount(1000, null, null), 1000);
  assert.equal(trainedRowCount(1000, "100", null), 900);
  assert.equal(trainedRowCount(1000, null, "99"), 100);
  assert.equal(trainedRowCount(1000, "10", "19"), 10);
  assert.equal(trainedRowCount(1000, " ", "5000"), 1000);
  assert.equal(trainedRowCount(1000, "abc", "-1"), 1000);
  assert.equal(trainedRowCount(1000, "2000", null), null);
  assert.equal(trainedRowCount(1000, "50", "10"), null);
  assert.equal(trainedRowCount(null), null);
  assert.equal(trainedRowCount(0), null);
});

test("the stored default is untouched: guidance only", () => {
  assert.equal(DEFAULT_HYPERPARAMS.maxSteps, 60);
  const hook = readSrc(
    "features/training/hooks/use-max-steps-epochs-toggle.ts",
  );
  assert.match(hook, /const DEFAULT_MAX_STEPS = 60;/);
});

test("the row count comes from the dataset check and is never persisted", () => {
  const store = readSrc("features/training/stores/training-config-store.ts");
  assert.match(
    store,
    /datasetTotalRows:\s*typeof res\.total_rows === "number" && res\.total_rows > 0/,
  );
  const persistence = readSrc(
    "features/training/stores/training-config-persistence.ts",
  );
  assert.match(persistence, /"datasetTotalRows",/);
});

test("the Max Steps field shows the hint and applies it only on click", () => {
  const params = readSrc("features/studio/sections/params-section.tsx");
  assert.match(params, /data-testid="llm-step-guidance"/);
  assert.match(params, /llmStepGuidance\(\s*trainedRowCount\(/);
  assert.match(
    params,
    /onClick=\{\(\) =>\s*store\.setMaxSteps\(stepGuidance\.suggestedSteps\)/,
  );
  assert.equal(params.match(/setMaxSteps\(stepGuidance/g)?.length, 1);
});

test("data-parallel replicas multiply the per-device batch", () => {
  // 10k rows, 4 x 8 per device on 4 DDP replicas = 128 rows per step.
  const ddp = llmStepGuidance(10_000, 4, 8, 4);
  assert.equal(ddp?.effectiveBatch, 128);
  assert.equal(ddp?.passSteps, 79);
  assert.equal(llmStepGuidance(10_000, 4, 8, 1)?.passSteps, 313);
  assert.equal(llmStepGuidance(10_000, 4, 8)?.passSteps, 313);
  // An unknown replica count withholds the hint instead of overestimating it.
  assert.equal(llmStepGuidance(10_000, 4, 8, null), null);
  assert.equal(llmStepGuidance(10_000, 4, 8, 0), null);
  assert.equal(llmStepGuidance(10_000, 4, 8, Number.NaN), null);
});

test("plan replicas mirror parallel_launch.data_parallel_size", () => {
  const plan = (mode: string, worldSize: number, spec: string | null) =>
    // biome-ignore lint/suspicious/noExplicitAny: test fixtures name every mode.
    planDataParallelSize({ mode: mode as any, worldSize, spec });
  assert.equal(plan("single", 1, null), 1);
  assert.equal(plan("layer_split", 1, null), 1);
  assert.equal(plan("ddp", 4, "dp=4"), 4);
  assert.equal(plan("fsdp2", 2, "dp=2,fsdp=1"), 2);
  assert.equal(plan("tp", 4, "tp=4"), 1);
  assert.equal(plan("pp", 2, "pp=2"), 1);
  assert.equal(plan("cp", 2, "cp=2"), 1);
  assert.equal(plan("ep", 4, "ep=4"), 4);
  assert.equal(plan("custom", 8, "dp=2;tp=2,pp=2"), 2);
  assert.equal(plan("custom", 4, "DP=2, TP_SIZE=2"), 2);
  // A multi-process plan without a readable layout is unknown, not 1.
  assert.equal(plan("ddp", 4, null), null);
  assert.equal(plan("custom", 4, " "), null);
  assert.equal(plan("custom", 4, "tp=two"), null);
  assert.equal(plan("ddp", 1, "dp=1"), 1);
});

test("the run's replicas: 1 off multi-GPU hosts, the plan's on one, else unknown", () => {
  const ddp = { mode: "ddp" as const, worldSize: 2, spec: "dp=2" };
  // Single-GPU host: capabilities are never fetched and /start sends no parallel fields.
  assert.equal(
    runDataParallelSize({
      capabilitiesStatus: "idle",
      gpuCount: null,
      plan: null,
    }),
    1,
  );
  assert.equal(
    runDataParallelSize({
      capabilitiesStatus: "ready",
      gpuCount: 1,
      plan: null,
    }),
    1,
  );
  // No multi-GPU routes on the backend: one process.
  for (const status of ["unsupported", "error"]) {
    assert.equal(
      runDataParallelSize({
        capabilitiesStatus: status,
        gpuCount: null,
        plan: null,
      }),
      1,
    );
  }
  assert.equal(
    runDataParallelSize({
      capabilitiesStatus: "loading",
      gpuCount: null,
      plan: null,
    }),
    null,
  );
  assert.equal(
    runDataParallelSize({
      capabilitiesStatus: "ready",
      gpuCount: 2,
      plan: ddp,
    }),
    2,
  );
  // Multi-GPU host with no plan yet, or a refused one: withheld.
  assert.equal(
    runDataParallelSize({
      capabilitiesStatus: "ready",
      gpuCount: 2,
      plan: null,
    }),
    null,
  );
  assert.equal(
    runDataParallelSize({
      capabilitiesStatus: "ready",
      gpuCount: 2,
      plan: { ...ddp, error: { code: "x", message: "no", alternative: null } },
    }),
    null,
  );
  assert.equal(
    runDataParallelSize({
      capabilitiesStatus: "ready",
      gpuCount: 4,
      plan: { mode: "single", worldSize: 1, spec: null },
    }),
    1,
  );
});

test("the Max Steps hint takes the run's data-parallel size", () => {
  const params = readSrc("features/studio/sections/params-section.tsx");
  assert.match(
    params,
    /useTrainingParallelStore\(\(s\) =>\s*runDataParallelSize\(/,
  );
  assert.match(params, /store\.gradientAccumulation,\s*dataParallelSize,\s*\)/);
});

test("a plan for an older GPU selection gives no replicas until the new plan lands", () => {
  const ddp4 = { mode: "ddp" as const, worldSize: 4, spec: "dp=4" };
  const base = { capabilitiesStatus: "ready", gpuCount: 4, plan: ddp4 };
  assert.equal(runDataParallelSize({ ...base, planFresh: true }), 4);
  assert.equal(runDataParallelSize({ ...base, planFresh: false }), null);
  // Single-GPU hosts never wait on a plan.
  assert.equal(
    runDataParallelSize({
      capabilitiesStatus: "ready",
      gpuCount: 1,
      plan: null,
      planFresh: false,
    }),
    1,
  );
});

test("plan freshness is keyed by the request, marked before the debounce", async () => {
  const hook = await readFile(
    new URL(
      "../src/features/training/hooks/use-parallel-plan.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const wanted = hook.indexOf("store.setWantedPlanKey(requestKey)");
  assert.ok(wanted > 0 && wanted < hook.indexOf("setTimeout("));
  assert.match(hook, /s\.setPlan\(result\.value, requestKey\)/);
  const section = await readFile(
    new URL(
      "../src/features/studio/sections/params-section.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(section, /s\.planRequestKey === s\.wantedPlanKey/);
  assert.match(section, /s\.planStatus === "ready"/);
});
