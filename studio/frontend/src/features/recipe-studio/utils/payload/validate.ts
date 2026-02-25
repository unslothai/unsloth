import type { ModelConfig, ModelProviderConfig, NodeConfig } from "../../types";

export function validateSubcategoryConfigs(
  configs: Record<string, NodeConfig>,
  nameToConfig: Map<string, NodeConfig>,
  errors: string[],
): void {
  for (const config of Object.values(configs)) {
    if (config.kind !== "sampler" || config.sampler_type !== "subcategory") {
      continue;
    }
    const parentName = config.subcategory_parent;
    if (!parentName) {
      errors.push(`Subcategory ${config.name}: parent category required.`);
      continue;
    }
    const parent = nameToConfig.get(parentName);
    const parentValues =
      parent && parent.kind === "sampler" && parent.sampler_type === "category"
        ? (parent.values ?? [])
        : [];
    const mapping = config.subcategory_mapping ?? {};
    for (const value of parentValues) {
      const list = mapping[value];
      if (!list || list.length === 0) {
        errors.push(
          `Subcategory ${config.name}: '${value}' needs at least 1 subcategory.`,
        );
      }
    }
  }
}

export function validateTimedeltaConfigs(
  configs: Record<string, NodeConfig>,
  nameToConfig: Map<string, NodeConfig>,
  errors: string[],
): void {
  for (const config of Object.values(configs)) {
    if (config.kind !== "sampler" || config.sampler_type !== "timedelta") {
      continue;
    }
    const reference = config.reference_column_name?.trim() ?? "";
    if (!reference) {
      errors.push(`Timedelta ${config.name}: reference datetime column required.`);
      continue;
    }
    const parent = nameToConfig.get(reference);
    if (
      !parent ||
      parent.kind !== "sampler" ||
      parent.sampler_type !== "datetime"
    ) {
      errors.push(`Timedelta ${config.name}: reference '${reference}' must be datetime.`);
    }
  }
}

export function validateModelAliasLinks(
  modelAliases: Set<string>,
  modelConfigConfigs: ModelConfig[],
  errors: string[],
): void {
  for (const alias of modelAliases) {
    if (!modelConfigConfigs.some((config) => config.name === alias)) {
      errors.push(`LLM model_alias ${alias}: missing model config.`);
    }
  }
}

export function validateModelConfigProviders(
  modelConfigConfigs: ModelConfig[],
  modelAliases: Set<string>,
  modelProviderNames: Set<string>,
  errors: string[],
): void {
  for (const config of modelConfigConfigs) {
    const provider = config.provider.trim();
    const alias = config.name;
    if (modelAliases.has(alias) && !config.model.trim()) {
      errors.push(`Model config ${alias}: model is required.`);
    }
    if (provider && !modelProviderNames.has(provider)) {
      errors.push(`Model config ${alias}: provider ${provider} not found.`);
    }
  }
}

export function validateUsedProviders(
  modelProviderConfigs: ModelProviderConfig[],
  modelConfigConfigs: ModelConfig[],
  errors: string[],
): void {
  const usedProviders = new Set(
    modelConfigConfigs.map((config) => config.provider.trim()).filter(Boolean),
  );
  for (const provider of modelProviderConfigs) {
    if (!usedProviders.has(provider.name)) {
      continue;
    }
    if (!provider.endpoint.trim()) {
      errors.push(`Model provider ${provider.name}: endpoint is required.`);
    }
    if (!provider.provider_type.trim()) {
      errors.push(`Model provider ${provider.name}: provider_type is required.`);
    }
  }
}
