// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listRecipeExecutions } from "../data/executions-db";
import type { RecipeExecutionRecord } from "../execution-types";
import {
  isExecutionInProgress,
  sortExecutions,
  withExecutionDefaults,
} from "./execution-helpers";

export async function loadSortedRecipeExecutions(
  recipeId: string,
): Promise<RecipeExecutionRecord[]> {
  const records = await listRecipeExecutions(recipeId);
  return sortExecutions(
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
