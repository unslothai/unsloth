export type RecipeStudioView = "editor" | "executions";

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
  status: RecipeExecutionStatus;
  rows: number;
  createdAt: number;
  finishedAt: number | null;
  recipeSignature: string;
  stage: string | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  current_column: string | null;
  progress: RecipeExecutionProgress | null;
  // biome-ignore lint/style/useNamingConvention: backend schema
  column_progress: RecipeExecutionProgress | null;
  batch: RecipeExecutionBatch | null;
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
