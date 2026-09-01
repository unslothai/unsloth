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
  // biome-ignore lint/style/useNamingConvention: backend schema
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
  // biome-ignore lint/style/useNamingConvention: backend schema
  page_items?: number | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  fetched_items?: number | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  estimated_total?: number | null;
  percent?: number | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  rate_remaining?: number | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  retry_after_sec?: number | null;
  message?: string | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  updated_at?: number | null;
};

export type RecipeExecutionAnalysis = {
  num_records?: number;
  target_num_records?: number;
  // biome-ignore lint/style/useNamingConvention: backend schema
  column_statistics?: Record<string, unknown>[];
  // biome-ignore lint/style/useNamingConvention: backend schema
  side_effect_column_names?: string[] | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  column_profiles?: Record<string, unknown>[] | null;
} & Record<string, unknown>;

export type RecipeExecutionRecord = {
  id: string;
  recipeId: string;
  // biome-ignore lint/style/useNamingConvention: backend schema
  jobId: string | null;
  kind: RecipeExecutionKind;
  // ui-only display label for full runs
  // biome-ignore lint/style/useNamingConvention: ui schema
  run_name: string | null;
  status: RecipeExecutionStatus;
  rows: number;
  createdAt: number;
  finishedAt: number | null;
  recipeSignature: string;
  stage: string | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  current_column: string | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  completed_columns: string[];
  progress: RecipeExecutionProgress | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  column_progress: RecipeExecutionProgress | null;
  batch: RecipeExecutionBatch | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  source_progress: RecipeSourceProgress | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  model_usage: Record<string, unknown> | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  lastEventId: number | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  artifact_path: string | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  log_lines: string[];
  dataset: Record<string, unknown>[];
  datasetTotal: number;
  datasetPage: number;
  datasetPageSize: number;
  analysis: RecipeExecutionAnalysis | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  processor_artifacts: Record<string, unknown> | null;
  error: string | null;
};
