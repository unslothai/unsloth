// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
export type ValidatorCodeLang =
  | "javascript"
  | "typescript"
  | "jsx"
  | "tsx"
  | "python"
  | "sql:sqlite"
  | "sql:postgres"
  | "sql:mysql"
  | "sql:tsql"
  | "sql:bigquery"
  | "sql:ansi";
export type ValidatorType = "code" | "oxc";
export type OxcValidationMode = "syntax" | "lint" | "syntax+lint";
export type OxcCodeShape = "auto" | "module" | "snippet";

export type ExpressionDtype = "str" | "int" | "float" | "bool";

export type LayoutDirection = "LR" | "TB";

export type SeedSamplingStrategy = "ordered" | "shuffle";
export type SeedSelectionType = "none" | "index_range" | "partition_block";
export type SeedSourceType = "hf" | "local" | "unstructured" | "github_repo";

export type GithubItemType = "issues" | "pulls" | "commits";
export type GithubStateFilter = "all" | "open" | "closed";
export const INFRA_NODE_KINDS = new Set([
  "model_provider",
  "model_config",
  "tool_config",
]);

export type RecipeNodeData = {
  title: string;
  name: string;
  kind:
    | "sampler"
    | "llm"
    | "validator"
    | "expression"
    | "seed"
    | "note"
    | "evaluation"
    | "model_provider"
    | "model_config"
    | "tool_config";
  subtype: string;
  blockType:
    | SamplerType
    | LlmType
    | "validator_python"
    | "validator_sql"
    | "validator_oxc"
    | "expression"
    | "seed"
    | "markdown_note"
    | "evaluation_document_score"
    | "model_provider"
    | "model_config"
    | "tool_config";
  layoutDirection?: LayoutDirection;
  runtimeState?: "idle" | "running" | "done";
  executionLocked?: boolean;
};

export type RecipeNode = Node<RecipeNodeData, "builder">;

export type CategoryConditionalParams = {
  sampler_type: "category";
  values: string[];
  weights?: Array<number | null>;
};

export type SamplerConfig = {
  id: string;
  kind: "sampler";
  // ui-only
  advancedOpen?: boolean;
  sampler_type: SamplerType;
  name: string;
  drop?: boolean;
  convert_to?: "float" | "int" | "str";
  values?: string[];
  weights?: Array<number | null>;
  low?: string;
  high?: string;
  mean?: string;
  std?: string;
  p?: string;
  datetime_start?: string;
  datetime_end?: string;
  datetime_unit?: string;
  dt_min?: string;
  dt_max?: string;
  reference_column_name?: string;
  timedelta_unit?: "D" | "h" | "m" | "s";
  uuid_format?: string;
  person_locale?: string;
  person_sex?: string;
  person_age_range?: string;
  person_city?: string;
  person_with_synthetic_personas?: boolean;
  subcategory_parent?: string;
  subcategory_mapping?: Record<string, string[]>;
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
  provider_type: McpProviderType;
  command?: string;
  args?: string[];
  env?: McpEnvVar[];
  endpoint?: string;
  api_key?: string;
  api_key_env?: string;
};

export type LlmToolConfig = {
  id: string;
  tool_alias: string;
  providers: string[];
  allow_tools?: string[];
  max_tool_call_turns?: string;
  timeout_sec?: string;
};

export type ToolProfileConfig = {
  id: string;
  kind: "tool_config";
  name: string;
  mcp_providers: LlmMcpProviderConfig[];
  fetched_tools_by_provider?: Record<string, string[]>;
  allow_tools?: string[];
  max_tool_call_turns?: string;
  timeout_sec?: string;
};

export type LlmImageContextConfig = {
  enabled: boolean;
  column_name: string;
};

export type LlmTraceType = "none" | "last_message" | "all_messages";

export type LlmConfig = {
  id: string;
  kind: "llm";
  // ui-only
  advancedOpen?: boolean;
  llm_type: LlmType;
  name: string;
  drop?: boolean;
  model_alias: string;
  prompt: string;
  system_prompt: string;
  code_lang?: string;
  output_format?: string;
  tool_alias?: string;
  scores?: Score[];
  // ui-only, serialized into multi_modal_context for DataDesigner
  image_context?: LlmImageContextConfig;
  with_trace?: LlmTraceType;
  extract_reasoning_content?: boolean;
};

export type ModelProviderConfig = {
  id: string;
  kind: "model_provider";
  name: string;
  endpoint: string;
  provider_type: string;
  api_key_env?: string;
  api_key?: string;
  extra_headers?: string;
  extra_body?: string;
  is_local?: boolean;
};

export type ModelConfig = {
  id: string;
  kind: "model_config";
  name: string;
  model: string;
  gguf_variant?: string;
  provider: string;
  inference_temperature?: string;
  inference_top_p?: string;
  inference_max_tokens?: string;
  inference_timeout?: string;
  inference_extra_body?: string;
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

export type ValidatorConfig = {
  id: string;
  kind: "validator";
  // ui-only
  advancedOpen?: boolean;
  name: string;
  drop?: boolean;
  target_columns: string[];
  // ui-only
  validator_type: ValidatorType;
  code_lang: ValidatorCodeLang;
  // ui-only (used for OXC validators)
  oxc_validation_mode: OxcValidationMode;
  // ui-only (used for OXC validators)
  oxc_code_shape: OxcCodeShape;
  // ui ergonomics (serialized to int in payload)
  batch_size: string;
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
  // ui-only
  advancedOpen?: boolean;
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
  unstructured_file_ids?: string[];
  unstructured_file_names?: string[];
  unstructured_file_sizes?: number[];
  github_repo_slug?: string;
  github_token?: string;
  github_limit?: string;
  github_item_types?: GithubItemType[];
  github_state?: GithubStateFilter;
  github_include_comments?: boolean;
  github_max_comments_per_item?: string;
  resolved_paths?: string[];
  // ui-only
  seed_preview_rows?: Record<string, unknown>[];
  // ui-only (string for input ergonomics)
  unstructured_chunk_size?: string;
  // ui-only (string for input ergonomics)
  unstructured_chunk_overlap?: string;
  seed_splits?: string[];
  // ui-only
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
  processor_type: "schema_transform";
  name: string;
  template: string;
};

export type EvaluationDocumentScoreConfig = {
  id: string;
  kind: "evaluation";
  evaluation_type: "json_document_score";
  name: string;
  prediction_column: string;
  reference_column: string;
  // JSON Schema or studio field-comparator map; raw text so the user can paste.
  schema: string;
  default_comparator: string;
  score_column: string;
  breakdown_column: string;
};

export type RecipeProcessorConfig = SchemaTransformProcessorConfig;

export type NodeConfig =
  | SamplerConfig
  | LlmConfig
  | ValidatorConfig
  | ExpressionConfig
  | MarkdownNoteConfig
  | SeedConfig
  | ModelProviderConfig
  | ModelConfig
  | ToolProfileConfig
  | EvaluationDocumentScoreConfig;
