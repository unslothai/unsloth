// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getRecipeJobDataset } from "../api";
import { listRecipeExecutionPage } from "../data/executions-db";
import type { RecipeExecutionRecord } from "../execution-types";
import {
  DATASET_PAGE_SIZE,
  isExecutionInProgress,
  normalizeDatasetRows,
  sortExecutions,
  withExecutionDefaults,
} from "./execution-helpers";

export async function hydrateCompletedFullExecutionDataset(
  record: RecipeExecutionRecord,
): Promise<RecipeExecutionRecord> {
  const execution = withExecutionDefaults(record);
  // Preview rows are not recoverable; restore only the first job-backed full-run page.
  if (
    execution.kind !== "full" ||
    execution.status !== "completed" ||
    !execution.jobId
  ) {
    return execution;
  }
  const response = await getRecipeJobDataset(execution.jobId, {
    limit: DATASET_PAGE_SIZE,
    offset: 0,
  });
  return {
    ...execution,
    dataset: normalizeDatasetRows(response.dataset),
    datasetTotal:
      typeof response.total === "number"
        ? response.total
        : execution.datasetTotal,
    datasetPage: 1,
    datasetPageSize: DATASET_PAGE_SIZE,
  };
}

export async function loadSortedRecipeExecutionPage(
  recipeId: string,
  cursor?: string | null,
): Promise<{
  executions: RecipeExecutionRecord[];
  nextCursor: string | null;
}> {
  const page = await listRecipeExecutionPage(recipeId, { cursor });
  const records = [...page.executions];
  if (
    !cursor &&
    page.resumable &&
    !records.some((record) => record.id === page.resumable?.id)
  ) {
    records.push(page.resumable);
  }
  const executions = sortExecutions(
    records.map((record) =>
      withExecutionDefaults({
        ...record,
        artifact_path: null,
        log_lines: [],
        dataset: [],
        datasetPage: 1,
        datasetPageSize: 20,
        processor_artifacts: null,
      }),
    ),
  );

  return {
    executions,
    nextCursor: page.nextCursor,
  };
}

export function findResumableExecution(
  records: RecipeExecutionRecord[],
): RecipeExecutionRecord | null {
  return (
    records.find(
      (record) => record.jobId && isExecutionInProgress(record.status),
    ) ?? null
  );
}
