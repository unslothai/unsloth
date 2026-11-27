// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ParallelMode } from "../types/parallel.ts";
import { showsGpuSection } from "./parallel-plan.ts";

// Fewer steps than this barely moves a LoRA, so tiny sets get a few passes instead of one.
export const MIN_SUGGESTED_STEPS = 30;
// Past this a first run is long enough to judge; train on part of a big set and raise it later.
export const MAX_SUGGESTED_STEPS = 3000;

export type LlmStepGuidance = {
  /** Rows the run trains on (the dataset slice when one is set). */
  rows: number;
  /** Rows per optimizer step: batch size x gradient accumulation x data-parallel replicas. */
  effectiveBatch: number;
  /** Steps for one pass over `rows`. */
  passSteps: number;
  /** passSteps kept within MIN_SUGGESTED_STEPS and MAX_SUGGESTED_STEPS. */
  suggestedSteps: number;
  kind: "pass" | "floor" | "cap";
};

function sliceBound(value: string | null | undefined): number | null {
  if (value == null || value.trim() === "") return null;
  const num = Number(value.trim());
  return Number.isInteger(num) && num >= 0 ? num : null;
}

/** Rows a run sees after the train split's inclusive [start, end] slice; null when the count is unknown. */
export function trainedRowCount(
  totalRows: number | null | undefined,
  sliceStart?: string | null,
  sliceEnd?: string | null,
): number | null {
  if (totalRows == null || !(totalRows > 0)) return null;
  const start = sliceBound(sliceStart) ?? 0;
  const endExclusive = Math.min(
    totalRows,
    (sliceBound(sliceEnd) ?? totalRows - 1) + 1,
  );
  const rows = endExclusive - start;
  return rows > 0 ? rows : null;
}

type PlanLayout = {
  mode: ParallelMode;
  worldSize: number;
  spec: string | null;
};

const MODEL_PARALLEL_AXES = new Set([
  "tp",
  "pp",
  "cp",
  "tp_size",
  "pp_size",
  "cp_size",
]);

/** Data-parallel replicas of a planned run, as the trainer counts them (parallel_launch.data_parallel_size:
 * world size over the tp x pp x cp ranks of the layout spec); null when the layout cannot say. */
export function planDataParallelSize(plan: PlanLayout): number | null {
  if (plan.mode === "single" || plan.mode === "layer_split") return 1;
  const world = Math.floor(plan.worldSize);
  if (!(world >= 1)) return null;
  if (world === 1) return 1;
  // Every multi-process plan carries its spec (it becomes UNSLOTH_PARALLEL); without one the layout is unknown.
  if (plan.spec == null || plan.spec.trim() === "") return null;
  let modelRanks = 1;
  for (const part of plan.spec.replaceAll(";", ",").split(",")) {
    const [rawKey, ...rest] = part.split("=");
    const key = rawKey.trim().toLowerCase();
    if (!MODEL_PARALLEL_AXES.has(key)) continue;
    const value = Number(rest.join("=").trim());
    if (!Number.isInteger(value)) return null;
    modelRanks *= Math.max(1, value);
  }
  return Math.max(1, Math.floor(world / modelRanks));
}

/** Data-parallel replicas for the run the Train page would start: 1 on a host with fewer than two
 * training GPUs (nothing parallel is sent), the plan's replicas on a multi-GPU host, and null while
 * that is not known (capabilities or the plan still loading, failed, or the layout unreadable). */
export function runDataParallelSize(state: {
  capabilitiesStatus: string;
  gpuCount: number | null;
  plan: (PlanLayout & { error?: unknown }) | null;
  /** False while the plan answers an older selection (debounced or loading); no hint then. */
  planFresh?: boolean;
}): number | null {
  if (state.gpuCount == null) {
    return state.capabilitiesStatus === "loading" ? null : 1;
  }
  if (!showsGpuSection(state.gpuCount)) return 1;
  if (!state.plan || state.plan.error || state.planFresh === false) return null;
  return planDataParallelSize(state.plan);
}

/** One pass is ceil(rows / (batch x grad accum x data-parallel replicas)) steps; the suggestion clamps
 * it to a sensible range. `batchSize` is per device, so an unknown replica count (null) gives no hint. */
export function llmStepGuidance(
  rows: number | null | undefined,
  batchSize: number,
  gradientAccumulation: number,
  dataParallelSize: number | null = 1,
): LlmStepGuidance | null {
  if (rows == null || !(rows > 0)) return null;
  if (dataParallelSize == null || !(dataParallelSize >= 1)) return null;
  const effectiveBatch =
    Math.max(1, Math.floor(batchSize) || 1) *
    Math.max(1, Math.floor(gradientAccumulation) || 1) *
    Math.floor(dataParallelSize);
  const passSteps = Math.ceil(rows / effectiveBatch);
  if (passSteps < MIN_SUGGESTED_STEPS) {
    return {
      rows,
      effectiveBatch,
      passSteps,
      suggestedSteps: MIN_SUGGESTED_STEPS,
      kind: "floor",
    };
  }
  if (passSteps > MAX_SUGGESTED_STEPS) {
    return {
      rows,
      effectiveBatch,
      passSteps,
      suggestedSteps: MAX_SUGGESTED_STEPS,
      kind: "cap",
    };
  }
  return {
    rows,
    effectiveBatch,
    passSteps,
    suggestedSteps: passSteps,
    kind: "pass",
  };
}
