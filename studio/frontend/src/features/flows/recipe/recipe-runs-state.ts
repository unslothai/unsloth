// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipeExecutionRecord } from "@/features/recipe-studio/execution-types";
import {
  isExecutionInProgress,
  sortExecutions,
  withExecutionDefaults,
} from "@/features/recipe-studio/executions/execution-helpers";
import type { RecipeRunsTarget } from "../stores/ui-store";

/** The backend's records for a Recipe Dataset step, in the order Recipe Studio lists its own. */
export function toRunRecords(raw: unknown[]): RecipeExecutionRecord[] {
  return sortExecutions(
    raw.map((record) =>
      withExecutionDefaults(record as RecipeExecutionRecord),
    ),
  );
}

/** Preview is the newest run alone; Runs is all of them. */
export function visibleRuns(
  records: RecipeExecutionRecord[],
  view: RecipeRunsTarget["view"],
): RecipeExecutionRecord[] {
  if (view === "runs") return records;
  const newest = [...records].sort((a, b) => b.createdAt - a.createdAt)[0];
  return newest ? [newest] : [];
}

export function selectedRun(
  shown: RecipeExecutionRecord[],
  picked: string | null,
): string | null {
  return picked && shown.some((r) => r.id === picked)
    ? picked
    : (shown[0]?.id ?? null);
}

export function runsAreLive(
  records: RecipeExecutionRecord[],
  flowRunning: boolean,
): boolean {
  return flowRunning || records.some((r) => isExecutionInProgress(r.status));
}
