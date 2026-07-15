// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listRecipeExecutionPage } from "../data/executions-db";
import type { RecipeExecutionRecord } from "../execution-types";
import {
  isExecutionInProgress,
  sortExecutions,
  withExecutionDefaults,
} from "./execution-helpers";

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
  return {
    executions: sortExecutions(
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
    ),
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
