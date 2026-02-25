import type { NodeConfig, SamplerConfig } from "../types";

export type DialogOptions = {
  categoryOptions: SamplerConfig[];
  modelConfigAliases: string[];
  modelProviderOptions: string[];
  datetimeOptions: string[];
};

export function buildDialogOptions(configList: NodeConfig[]): DialogOptions {
  const categoryOptions: SamplerConfig[] = [];
  const modelConfigAliases: string[] = [];
  const modelProviderOptions: string[] = [];
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
    }
  }

  return {
    categoryOptions,
    modelConfigAliases,
    modelProviderOptions,
    datetimeOptions,
  };
}
