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
    raw.map((record) => withExecutionDefaults(record as RecipeExecutionRecord)),
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

/** A finished run's saved parquet as a flow `dataset` value, so its pages outlive the recipe job. */
export function savedRunDataset(
  record: Pick<RecipeExecutionRecord, "artifact_path" | "output_ref">,
): { kind: "dataset"; source: "recipe"; ref: string } | null {
  // A schema transform's rows are the run's output; they are saved beside the job's own.
  if (record.output_ref)
    return { kind: "dataset", source: "recipe", ref: record.output_ref };
  const name = record.artifact_path
    ?.trim()
    .split(/[\\/]/)
    .filter(Boolean)
    .pop();
  return name
    ? {
        kind: "dataset",
        source: "recipe",
        ref: `recipes/${name}/parquet-files`,
      }
    : null;
}

/** One page of a full run's rows. The recipe job manager keeps only the newest job, so an older run
 * pages from its saved parquet; a schema transform's pages always read its own output, since the
 * job's dataset is the untransformed one. */
export async function readRunPage(
  record: Pick<RecipeExecutionRecord, "jobId" | "artifact_path" | "output_ref">,
  offset: number,
  size: number,
  read: {
    job: (
      jobId: string,
      page: { limit: number; offset: number },
    ) => Promise<{ dataset?: unknown; total?: unknown }>;
    saved: (
      value: NonNullable<ReturnType<typeof savedRunDataset>>,
      offset: number,
      limit: number,
    ) => Promise<{ rows: unknown; total: number | null; view_rows: number }>;
  },
): Promise<{ dataset?: unknown; total?: unknown }> {
  const saved = savedRunDataset(record);
  const fromJob =
    record.output_ref || !record.jobId
      ? Promise.reject(new Error("transformed output"))
      : read.job(record.jobId, { limit: size, offset });
  return fromJob.catch(async (error) => {
    if (!saved) throw error;
    const page = await read.saved(saved, offset, size);
    // That reader stops at its view window; pages past it would load empty.
    const total =
      typeof page.total === "number"
        ? Math.min(page.total, page.view_rows)
        : page.total;
    return { dataset: page.rows, total };
  });
}
