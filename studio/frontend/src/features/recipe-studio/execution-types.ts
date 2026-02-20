export type RecipeStudioView = "editor" | "executions";

export type RecipeExecutionKind = "preview" | "full";

export type RecipeExecutionStatus = "running" | "completed" | "error";

export type RecipeExecutionRecord = {
  id: string;
  recipeId: string;
  kind: RecipeExecutionKind;
  status: RecipeExecutionStatus;
  rows: number;
  createdAt: number;
  recipeSignature: string;
  dataset: Record<string, unknown>[];
  analysis: Record<string, unknown> | null;
  // biome-ignore lint/style/useNamingConvention: api schema
  processor_artifacts: Record<string, unknown> | null;
  error: string | null;
};
