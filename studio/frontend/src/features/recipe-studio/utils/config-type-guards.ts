import type {
  ExpressionConfig,
  LlmConfig,
  NodeConfig,
  SamplerConfig,
} from "../types";

export function isSamplerConfig(
  config: NodeConfig | null | undefined,
): config is SamplerConfig {
  return Boolean(config && config.kind === "sampler");
}

export function isCategoryConfig(
  config: NodeConfig | null | undefined,
): config is SamplerConfig {
  return Boolean(
    config && config.kind === "sampler" && config.sampler_type === "category",
  );
}

export function isSubcategoryConfig(
  config: NodeConfig | null | undefined,
): config is SamplerConfig {
  return Boolean(
    config &&
      config.kind === "sampler" &&
      config.sampler_type === "subcategory",
  );
}

export function isLlmConfig(
  config: NodeConfig | null | undefined,
): config is LlmConfig {
  return Boolean(config && config.kind === "llm");
}

export function isExpressionConfig(
  config: NodeConfig | null | undefined,
): config is ExpressionConfig {
  return Boolean(config && config.kind === "expression");
}
