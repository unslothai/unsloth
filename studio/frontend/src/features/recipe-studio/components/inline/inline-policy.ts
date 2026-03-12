// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig, SamplerType } from "../../types";

export type ConfigUiMode = "inline" | "dialog";

const INLINE_SAMPLERS = new Set<SamplerType>([
  "uniform",
  "gaussian",
  "bernoulli",
  "uuid",
]);

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
  if (config.kind === "tool_config") {
    return "dialog";
  }
  if (config.kind === "llm") {
    if (config.llm_type === "text" || config.llm_type === "code") {
      return "inline";
    }
    return "dialog";
  }
  if (config.kind === "seed") {
    return "inline";
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
