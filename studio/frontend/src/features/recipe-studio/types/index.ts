import type { Node } from "@xyflow/react";

export type SamplerType =
  | "category"
  | "subcategory"
  | "uniform"
  | "gaussian"
  | "bernoulli"
  | "datetime"
  | "timedelta"
  | "uuid"
  | "person"
  | "person_from_faker";

export type LlmType = "text" | "structured" | "code" | "judge";

export type ExpressionDtype = "str" | "int" | "float" | "bool";

export type LayoutDirection = "LR" | "TB";

export type SeedSamplingStrategy = "ordered" | "shuffle";
export type SeedSelectionType = "none" | "index_range" | "partition_block";
export type SeedSourceType = "hf" | "local" | "unstructured";

export type RecipeNodeData = {
  title: string;
  name: string;
  kind:
    | "sampler"
    | "llm"
    | "expression"
    | "seed"
    | "note"
    | "model_provider"
    | "model_config";
  subtype: string;
  blockType:
    | SamplerType
    | LlmType
    | "expression"
    | "seed"
    | "markdown_note"
    | "model_provider"
    | "model_config";
  layoutDirection?: LayoutDirection;
};

export type RecipeNode = Node<RecipeNodeData, "builder">;

export type CategoryConditionalParams = {
  // biome-ignore lint/style/useNamingConvention: api schema
  sampler_type: "category";
  values: string[];
  weights?: Array<number | null>;
};

export type SamplerConfig = {
  id: string;
  kind: "sampler";
  // biome-ignore lint/style/useNamingConvention: api schema
  sampler_type: SamplerType;
  name: string;
  drop?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  convert_to?: "float" | "int" | "str";
  values?: string[];
  weights?: Array<number | null>;
  low?: string;
  high?: string;
  mean?: string;
  std?: string;
  p?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  datetime_start?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  datetime_end?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  datetime_unit?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  dt_min?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  dt_max?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  reference_column_name?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  timedelta_unit?: "D" | "h" | "m" | "s";
  // biome-ignore lint/style/useNamingConvention: api schema
  uuid_format?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  person_locale?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  person_sex?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  person_age_range?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  person_city?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  person_with_synthetic_personas?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  subcategory_parent?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  subcategory_mapping?: Record<string, string[]>;
  // biome-ignore lint/style/useNamingConvention: api schema
  conditional_params?: Record<string, CategoryConditionalParams>;
};

export type ScoreOption = {
  value: string;
  description: string;
};

export type Score = {
  name: string;
  description: string;
  options: ScoreOption[];
};

export type McpProviderType = "stdio" | "streamable_http";

export type McpEnvVar = {
  key: string;
  value: string;
};

export type LlmMcpProviderConfig = {
  id: string;
  name: string;
  // biome-ignore lint/style/useNamingConvention: ui schema
  provider_type: McpProviderType;
  command?: string;
  args?: string[];
  env?: McpEnvVar[];
  endpoint?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  api_key?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  api_key_env?: string;
};

export type LlmToolConfig = {
  id: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  tool_alias: string;
  providers: string[];
  // biome-ignore lint/style/useNamingConvention: api schema
  allow_tools?: string[];
  // biome-ignore lint/style/useNamingConvention: api schema
  max_tool_call_turns?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  timeout_sec?: string;
};

export type LlmConfig = {
  id: string;
  kind: "llm";
  // biome-ignore lint/style/useNamingConvention: api schema
  llm_type: LlmType;
  name: string;
  drop?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  model_alias: string;
  prompt: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  system_prompt: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  code_lang?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  output_format?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  tool_alias?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  tool_configs?: LlmToolConfig[];
  // biome-ignore lint/style/useNamingConvention: ui schema
  mcp_providers?: LlmMcpProviderConfig[];
  scores?: Score[];
};

export type ModelProviderConfig = {
  id: string;
  kind: "model_provider";
  name: string;
  endpoint: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  provider_type: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  api_key_env?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  api_key?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  extra_headers?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  extra_body?: string;
};

export type ModelConfig = {
  id: string;
  kind: "model_config";
  name: string;
  model: string;
  provider: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  inference_temperature?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  inference_top_p?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  inference_max_tokens?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  skip_health_check?: boolean;
};

export type ExpressionConfig = {
  id: string;
  kind: "expression";
  name: string;
  drop?: boolean;
  expr: string;
  dtype: ExpressionDtype;
};

export type MarkdownNoteConfig = {
  id: string;
  kind: "markdown_note";
  name: string;
  markdown: string;
  // ui-only
  note_color?: string;
  // ui-only (0-100 as string for slider/input ergonomics)
  note_opacity?: string;
};

export type SeedConfig = {
  id: string;
  kind: "seed";
  name: string;
  drop?: boolean;
  // ui-only: explicit per-column drop for structured seed sources (hf/local)
  seed_drop_columns?: string[];
  seed_source_type: SeedSourceType;
  // ui-only (serialized in seed_config)
  hf_repo_id: string;
  hf_subset?: string;
  hf_split?: string;
  hf_path: string;
  hf_token?: string;
  hf_endpoint?: string;
  local_file_name?: string;
  unstructured_file_name?: string;
  // ui-only
  seed_preview_rows?: Record<string, unknown>[];
  // ui-only (string for input ergonomics)
  unstructured_chunk_size?: string;
  // ui-only (string for input ergonomics)
  unstructured_chunk_overlap?: string;
  seed_splits?: string[];
  // ui-only
  // biome-ignore lint/style/useNamingConvention: ui schema
  seed_globs_by_split?: Record<string, string>;
  seed_columns?: string[];
  sampling_strategy: SeedSamplingStrategy;
  selection_type: SeedSelectionType;
  selection_start?: string;
  selection_end?: string;
  selection_index?: string;
  selection_num_partitions?: string;
};

export type SchemaTransformProcessorConfig = {
  id: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  processor_type: "schema_transform";
  name: string;
  template: string;
};

export type RecipeProcessorConfig = SchemaTransformProcessorConfig;

export type NodeConfig =
  | SamplerConfig
  | LlmConfig
  | ExpressionConfig
  | MarkdownNoteConfig
  | SeedConfig
  | ModelProviderConfig
  | ModelConfig;
