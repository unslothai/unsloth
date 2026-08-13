import { getDefaultModels, listTenantModels } from "./model-api";
import type { PlatformDefaultModel, PlatformModel } from "./model-types";

export type PlatformReadinessRequirement = "chat" | "embedding" | "both";

export interface PlatformModelReadiness {
  defaults: PlatformDefaultModel[];
  missing: string[];
  models: PlatformModel[];
  ready: boolean;
}

const requiredCapabilities = (
  requirement: PlatformReadinessRequirement,
): string[] => (requirement === "both" ? ["chat", "embedding"] : [requirement]);

export function evaluatePlatformReadiness(
  models: PlatformModel[],
  defaults: PlatformDefaultModel[],
  requirement: PlatformReadinessRequirement,
): PlatformModelReadiness {
  const missing = requiredCapabilities(requirement).filter((capability) => {
    const selected = defaults.find(
      (item) => item.capability === capability && item.enabled,
    );
    if (!selected || (!selected.modelId && !selected.modelName)) return true;
    return !models.some(
      (model) =>
        model.capabilities.includes(capability) &&
        (model.id === selected.modelId || model.name === selected.modelName),
    );
  });
  return { defaults, missing, models, ready: missing.length === 0 };
}

export async function getPlatformModelReadiness(
  requirement: PlatformReadinessRequirement,
  signal?: AbortSignal,
): Promise<PlatformModelReadiness> {
  const [models, defaults] = await Promise.all([
    listTenantModels(signal),
    getDefaultModels(signal),
  ]);
  return evaluatePlatformReadiness(models, defaults, requirement);
}

export interface DatasetPipelineFields {
  parse_type: 2;
  pipeline_id: string;
}

export function mapPipelineToDatasetFields(
  pipelineId: string,
): DatasetPipelineFields | null {
  const normalized = pipelineId.trim();
  return normalized ? { pipeline_id: normalized, parse_type: 2 } : null;
}
