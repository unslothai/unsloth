// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig, SamplerConfig } from "../types";

export type DialogOptions = {
  categoryOptions: SamplerConfig[];
  modelConfigAliases: string[];
  modelProviderOptions: string[];
  localProviderNames: Set<string>;
  toolProfileAliases: string[];
  datetimeOptions: string[];
};

export function buildDialogOptions(configList: NodeConfig[]): DialogOptions {
  const categoryOptions: SamplerConfig[] = [];
  const modelConfigAliases: string[] = [];
  const modelProviderOptions: string[] = [];
  const localProviderNames = new Set<string>();
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
      if (config.is_local) {
        localProviderNames.add(config.name);
      }
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
    localProviderNames,
    toolProfileAliases,
    datetimeOptions,
  };
}
