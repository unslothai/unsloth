// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Fewer steps than this barely moves a LoRA, so tiny sets get a few passes instead of one.
export const MIN_SUGGESTED_STEPS = 30;
// Past this a first run is long enough to judge; train on part of a big set and raise it later.
export const MAX_SUGGESTED_STEPS = 3000;

export type LlmStepGuidance = {
  /** Rows the run trains on (the dataset slice when one is set). */
  rows: number;
  /** Rows per optimizer step: batch size x gradient accumulation. */
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

/** One pass is ceil(rows / (batch x grad accum)) steps; the suggestion clamps it to a sensible range. */
export function llmStepGuidance(
  rows: number | null | undefined,
  batchSize: number,
  gradientAccumulation: number,
): LlmStepGuidance | null {
  if (rows == null || !(rows > 0)) return null;
  const effectiveBatch =
    Math.max(1, Math.floor(batchSize) || 1) *
    Math.max(1, Math.floor(gradientAccumulation) || 1);
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
