import type { Node } from "@xyflow/react";

export type SamplerType =
  | "category"
  | "subcategory"
  | "uniform"
  | "gaussian"
  | "datetime"
  | "uuid"
  | "person"
  | "person_from_faker";

export type LlmType = "text" | "structured" | "code" | "judge";

export type ExpressionDtype = "str" | "int" | "float" | "bool";

export type LayoutDirection = "LR" | "TB";

export type CanvasNodeData = {
  title: string;
  name: string;
  kind: "sampler" | "llm" | "expression" | "model_provider" | "model_config";
  subtype: string;
  blockType:
    | SamplerType
    | LlmType
    | "expression"
    | "model_provider"
    | "model_config";
  layoutDirection?: LayoutDirection;
};

export type CanvasNode = Node<CanvasNodeData, "builder">;

export type SamplerConfig = {
  id: string;
  kind: "sampler";
  // biome-ignore lint/style/useNamingConvention: api schema
  sampler_type: SamplerType;
  name: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  convert_to?: "float" | "int" | "str";
  values?: string[];
  weights?: Array<number | null>;
  low?: string;
  high?: string;
  mean?: string;
  std?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  datetime_start?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  datetime_end?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  datetime_unit?: string;
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

export type LlmConfig = {
  id: string;
  kind: "llm";
  // biome-ignore lint/style/useNamingConvention: api schema
  llm_type: LlmType;
  name: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  model_alias: string;
  prompt: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  system_prompt: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  code_lang?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  output_format?: string;
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
  expr: string;
  dtype: ExpressionDtype;
};

export type NodeConfig =
  | SamplerConfig
  | LlmConfig
  | ExpressionConfig
  | ModelProviderConfig
  | ModelConfig;
