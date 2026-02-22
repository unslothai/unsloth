export type RecipePayload = {
  recipe: {
    // biome-ignore lint/style/useNamingConvention: api schema
    model_providers: Record<string, unknown>[];
    // biome-ignore lint/style/useNamingConvention: api schema
    mcp_providers: Record<string, unknown>[];
    // biome-ignore lint/style/useNamingConvention: api schema
    model_configs: Record<string, unknown>[];
    // biome-ignore lint/style/useNamingConvention: api schema
    seed_config?: Record<string, unknown>;
    // biome-ignore lint/style/useNamingConvention: api schema
    tool_configs: Record<string, unknown>[];
    columns: Record<string, unknown>[];
    processors: Record<string, unknown>[];
  };
  run: {
    rows: number;
    preview: boolean;
    // biome-ignore lint/style/useNamingConvention: api schema
    output_formats: string[];
    // biome-ignore lint/style/useNamingConvention: backend schema
    execution_type?: "preview" | "full";
    // biome-ignore lint/style/useNamingConvention: backend schema
    run_config?: Record<string, unknown>;
    // biome-ignore lint/style/useNamingConvention: backend schema
    dataset_name?: string;
    // biome-ignore lint/style/useNamingConvention: backend schema
    artifact_path?: string;
  };
  ui: {
    nodes: { id: string; x: number; y: number }[];
    edges: { from: string; to: string; type?: string }[];
  };
};

export type RecipePayloadResult = {
  errors: string[];
  payload: RecipePayload;
};
