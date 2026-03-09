// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type { NodeConfig, SamplerConfig } from "../types";

export type DialogOptions = {
  categoryOptions: SamplerConfig[];
  modelConfigAliases: string[];
  modelProviderOptions: string[];
  toolProfileAliases: string[];
  datetimeOptions: string[];
};

export function buildDialogOptions(configList: NodeConfig[]): DialogOptions {
  const categoryOptions: SamplerConfig[] = [];
  const modelConfigAliases: string[] = [];
  const modelProviderOptions: string[] = [];
  const toolProfileAliases: string[] = [];
  const datetimeOptions: string[] = [];

  for (const config of configList) {
    if (config.kind === "sampler") {
      if (config.sampler_type === "category") {
        categoryOptions.push(config);
      }
      if (config.sampler_type === "datetime") {
        datetimeOptions.push(config.name);
      }
      continue;
    }
    if (config.kind === "model_config") {
      modelConfigAliases.push(config.name);
      continue;
    }
    if (config.kind === "model_provider") {
      modelProviderOptions.push(config.name);
      continue;
    }
    if (config.kind === "tool_config") {
      toolProfileAliases.push(config.name);
    }
  }

  return {
    categoryOptions,
    modelConfigAliases,
    modelProviderOptions,
    toolProfileAliases,
    datetimeOptions,
  };
}
