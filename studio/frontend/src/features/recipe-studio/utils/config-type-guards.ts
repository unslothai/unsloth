// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ExpressionConfig,
  ImageConfig,
  LlmConfig,
  ModelConfig,
  NodeConfig,
  SamplerConfig,
  ValidatorConfig,
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

export function isImageConfig(
  config: NodeConfig | null | undefined,
): config is ImageConfig {
  return Boolean(config && config.kind === "image");
}

export function isImageModelConfig(
  config: NodeConfig | null | undefined,
): config is ModelConfig {
  return Boolean(
    config && config.kind === "model_config" && config.generation === "image",
  );
}

export function isValidatorConfig(
  config: NodeConfig | null | undefined,
): config is ValidatorConfig {
  return Boolean(config && config.kind === "validator");
}
