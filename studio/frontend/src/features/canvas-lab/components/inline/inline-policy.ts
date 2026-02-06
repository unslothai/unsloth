import type { NodeConfig, SamplerType } from "../../types";

export type ConfigUiMode = "inline" | "dialog";

type InlineFieldMap = Record<string, readonly string[]>;

const INLINE_SAMPLERS = new Set<SamplerType>([
  "uniform",
  "gaussian",
  "bernoulli",
  "uuid",
]);

const INLINE_FIELD_MAP: InlineFieldMap = {
  uniform: ["low", "high", "convert_to"],
  gaussian: ["mean", "std", "convert_to"],
  bernoulli: ["p"],
  uuid: ["uuid_format"],
  model_provider: ["provider_type", "endpoint"],
  model_config: ["provider", "model", "inference_temperature"],
  llm_text: ["model_alias"],
  llm_code: ["model_alias", "code_lang"],
  expression: ["dtype", "expr"],
};

export function getInlineFields(config: NodeConfig): readonly string[] {
  if (config.kind === "sampler") {
    return INLINE_FIELD_MAP[config.sampler_type] ?? [];
  }
  if (config.kind === "model_provider") {
    return INLINE_FIELD_MAP.model_provider;
  }
  if (config.kind === "model_config") {
    return INLINE_FIELD_MAP.model_config;
  }
  if (config.kind === "llm" && config.llm_type === "text") {
    return INLINE_FIELD_MAP.llm_text;
  }
  if (config.kind === "llm" && config.llm_type === "code") {
    return INLINE_FIELD_MAP.llm_code;
  }
  if (config.kind === "expression") {
    return INLINE_FIELD_MAP.expression;
  }
  return [];
}

export function getConfigUiMode(
  config: NodeConfig | null | undefined,
): ConfigUiMode {
  if (!config) {
    return "dialog";
  }
  if (config.kind === "sampler") {
    return INLINE_SAMPLERS.has(config.sampler_type) ? "inline" : "dialog";
  }
  if (config.kind === "model_provider" || config.kind === "model_config") {
    return "inline";
  }
  if (config.kind === "llm") {
    if (config.llm_type === "text" || config.llm_type === "code") {
      return "inline";
    }
    return "dialog";
  }
  if (config.kind === "expression") {
    return "inline";
  }
  return "dialog";
}

export function isInlineConfig(
  config: NodeConfig | null | undefined,
): boolean {
  return getConfigUiMode(config) === "inline";
}
