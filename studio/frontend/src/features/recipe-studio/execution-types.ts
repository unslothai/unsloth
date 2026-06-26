// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type RecipeStudioView = "easy" | "editor" | "executions";

export type RecipeExecutionKind = "preview" | "full";

export type RecipeExecutionStatus =
  | "pending"
  | "running"
  | "active"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "error";

export type RecipeExecutionProgress = {
  done?: number | null;
  total?: number | null;
  percent?: number | null;
  eta_sec?: number | null;
  rate?: number | null;
  ok?: number | null;
  failed?: number | null;
};

export type RecipeExecutionBatch = {
  idx?: number | null;
  total?: number | null;
};

export type RecipeSourceProgress = {
  source?: string | null;
  status?: string | null;
  repo?: string | null;
  resource?: string | null;
  page?: number | null;
  page_items?: number | null;
  fetched_items?: number | null;
  estimated_total?: number | null;
  percent?: number | null;
  rate_remaining?: number | null;
  retry_after_sec?: number | null;
  message?: string | null;
  updated_at?: number | null;
};

export type RecipeExecutionAnalysis = {
  num_records?: number;
  target_num_records?: number;
  column_statistics?: Record<string, unknown>[];
  side_effect_column_names?: string[] | null;
  column_profiles?: Record<string, unknown>[] | null;
} & Record<string, unknown>;

export type RecipeExecutionRecord = {
  id: string;
  recipeId: string;
  jobId: string | null;
  kind: RecipeExecutionKind;
  // ui-only display label for full runs
  run_name: string | null;
  status: RecipeExecutionStatus;
  rows: number;
  createdAt: number;
  finishedAt: number | null;
  recipeSignature: string;
  stage: string | null;
  current_column: string | null;
  completed_columns: string[];
  progress: RecipeExecutionProgress | null;
  column_progress: RecipeExecutionProgress | null;
  batch: RecipeExecutionBatch | null;
  source_progress: RecipeSourceProgress | null;
  model_usage: Record<string, unknown> | null;
  lastEventId: number | null;
  artifact_path: string | null;
  log_lines: string[];
  dataset: Record<string, unknown>[];
  datasetTotal: number;
  datasetPage: number;
  datasetPageSize: number;
  analysis: RecipeExecutionAnalysis | null;
  processor_artifacts: Record<string, unknown> | null;
  error: string | null;
};
