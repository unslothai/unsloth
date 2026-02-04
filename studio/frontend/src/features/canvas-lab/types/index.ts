import type { Node } from "@xyflow/react";

export type SamplerType =
  | "category"
  | "subcategory"
  | "uniform"
  | "gaussian"
  | "datetime"
  | "uuid"
  | "person";

export type LlmType = "text" | "structured" | "code";

export type CanvasNodeData = {
  title: string;
  name: string;
  kind: "sampler" | "llm";
  subtype: string;
};

export type CanvasNode = Node<CanvasNodeData, "builder">;

export type SamplerConfig = {
  id: string;
  kind: "sampler";
  // biome-ignore lint/style/useNamingConvention: api schema
  sampler_type: SamplerType;
  name: string;
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
  person_sample_dataset_when_available?: boolean;
  // biome-ignore lint/style/useNamingConvention: api schema
  subcategory_parent?: string;
  // biome-ignore lint/style/useNamingConvention: api schema
  subcategory_mapping?: Record<string, string[]>;
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
};

export type NodeConfig = SamplerConfig | LlmConfig;
