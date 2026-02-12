import type { NodeConfig, SamplerConfig } from "../types";

export type PreviewSummary = {
  totalColumns: number;
  llmColumns: number;
  samplerColumns: number;
  expressionColumns: number;
  toolConfigs: number;
  mcpProviders: number;
};

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

export function buildPreviewSummary(configList: NodeConfig[]): PreviewSummary {
  const toolConfigAliases = new Set<string>();
  const mcpProviderNames = new Set<string>();
  let totalColumns = 0;
  let llmColumns = 0;
  let samplerColumns = 0;
  let expressionColumns = 0;

  for (const config of configList) {
    if (config.kind === "sampler") {
      totalColumns += 1;
      samplerColumns += 1;
      continue;
    }
    if (config.kind === "expression") {
      totalColumns += 1;
      expressionColumns += 1;
      continue;
    }
    if (config.kind !== "llm") {
      continue;
    }

    totalColumns += 1;
    llmColumns += 1;
    for (const toolConfig of config.tool_configs ?? []) {
      const toolAlias = toolConfig.tool_alias.trim();
      if (toolAlias) {
        toolConfigAliases.add(toolAlias);
      }
    }

    for (const provider of config.mcp_providers ?? []) {
      const providerName = provider.name.trim();
      if (providerName) {
        mcpProviderNames.add(providerName);
      }
    }
  }

  return {
    totalColumns,
    llmColumns,
    samplerColumns,
    expressionColumns,
    toolConfigs: toolConfigAliases.size,
    mcpProviders: mcpProviderNames.size,
  };
}
