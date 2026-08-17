import type { PlatformTenantModels } from "./auth-types";

export interface PlatformProvider {
  id: string;
  name: string;
  description: string;
  hasInstance: boolean;
}

export interface PlatformProviderInstance {
  id: string;
  name: string;
  providerId: string;
  region: string;
  baseUrl: string;
  status: string;
  hasCredential: boolean;
}

export interface PlatformModel {
  id: string;
  name: string;
  providerId: string;
  providerName: string;
  instanceId: string;
  instanceName: string;
  capabilities: string[];
  status: string;
  maxTokens: number | null;
}

/**
 * Runtime model consumers resolve tenant models from the backend's composite
 * reference: model@provider for the default instance, or
 * model@instance@provider for a named instance. The model catalog keeps the
 * bare model id and provider metadata separate, so product screens compose the
 * reference before persisting memory/search configuration.
 */
export function platformModelReference(
  model: Pick<PlatformModel, "id" | "name" | "providerName" | "instanceName">,
): string {
  const name = model.name.trim() || model.id.trim();
  const provider = model.providerName.trim();
  const instance = model.instanceName.trim();
  if (!provider) return model.id.trim() || name;
  if (instance && instance.toLowerCase() !== "default") {
    return `${name}@${instance}@${provider}`;
  }
  return `${name}@${provider}`;
}

export function resolvePlatformModelReference(
  reference: string,
  models: PlatformModel[],
): string {
  const normalized = reference.trim();
  if (!normalized) return "";
  const match = models.find((model) => {
    const composite = platformModelReference(model);
    return (
      normalized === model.id.trim() ||
      normalized === model.name.trim() ||
      normalized === composite
    );
  });
  return match ? platformModelReference(match) : normalized;
}

export interface PlatformDefaultModel {
  capability: string;
  enabled: boolean;
  instanceName: string;
  modelId: string;
  modelName: string;
  providerName: string;
}

export interface PlatformPipeline {
  description: string;
  filename: string;
  id: string;
  title: string;
}

export interface PlatformProviderTask {
  id: string;
  status: string;
}

export interface PlatformTaskSegment {
  content: string;
  index: number;
}

export interface PlatformEmbeddingResult {
  index: number;
  tokenCount: number;
  vector: number[];
}

export interface PlatformRerankResult {
  index: number;
  relevanceScore: number;
}

type UnknownRecord = Record<string, unknown>;

export function asRecord(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

export function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function booleanValue(value: unknown): boolean {
  return value === true || value === 1 || value === "1" || value === "true";
}

function numberValue(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function stringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .filter((item): item is string => typeof item === "string")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return typeof value === "string"
    ? value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
    : [];
}

const MODEL_TYPE_CAPABILITIES = [
  [1, "chat"],
  [2, "embedding"],
  [4, "asr"],
  [8, "vision"],
  [16, "rerank"],
  [32, "tts"],
  [64, "ocr"],
] as const;

/**
 * The current Go model store represents model_type as a bitmask, while older
 * and catalog endpoints return a string or string array. Keep that transport
 * variation inside the adapter so chat and embedding consumers always receive
 * the same capability names.
 */
export function modelCapabilities(value: unknown): string[] {
  const values = Array.isArray(value)
    ? value
    : typeof value === "string"
      ? value.split(",")
      : [value];
  const labels: string[] = [];
  let numericMask = 0;
  for (const item of values) {
    const numericValue =
      typeof item === "number"
        ? item
        : typeof item === "string" && /^\d+$/.test(item.trim())
          ? Number(item)
          : null;
    if (numericValue !== null && Number.isSafeInteger(numericValue)) {
      numericMask |= numericValue;
      continue;
    }
    if (typeof item === "string" && item.trim()) labels.push(item.trim());
  }
  const decoded = MODEL_TYPE_CAPABILITIES.flatMap(([bit, capability]) =>
    (numericMask & bit) !== 0 ? [capability] : [],
  );
  return [...new Set([...labels, ...decoded])];
}

export function mapProvider(value: unknown): PlatformProvider | null {
  const dto = asRecord(value);
  const name = stringValue(dto.name || dto.provider_name).trim();
  if (!name) return null;
  return {
    id: stringValue(dto.id || dto.provider_id).trim() || name,
    name,
    description: stringValue(dto.description).trim(),
    hasInstance: booleanValue(dto.has_instance),
  };
}

export function mapProviderInstance(
  value: unknown,
): PlatformProviderInstance | null {
  const dto = asRecord(value);
  const name = stringValue(dto.instance_name || dto.name).trim();
  if (!name) return null;
  // api_key is deliberately reduced to a boolean and never enters the domain.
  const apiKey = dto.api_key;
  return {
    id: stringValue(dto.id).trim() || name,
    name,
    providerId: stringValue(dto.provider_id).trim(),
    region: stringValue(dto.region).trim(),
    baseUrl: stringValue(dto.base_url).trim(),
    status: stringValue(dto.status).trim(),
    hasCredential:
      (typeof apiKey === "string" && apiKey.length > 0) ||
      (typeof apiKey === "object" && apiKey !== null),
  };
}

export function mapModel(value: unknown): PlatformModel | null {
  const dto = asRecord(value);
  const name = stringValue(dto.name || dto.model_name).trim();
  const id = stringValue(dto.model_id || dto.id).trim() || name;
  if (!id || !name) return null;
  return {
    id,
    name,
    providerId: stringValue(dto.provider_id).trim(),
    providerName: stringValue(dto.provider_name).trim(),
    instanceId: stringValue(dto.instance_id).trim(),
    instanceName: stringValue(dto.instance_name).trim(),
    capabilities: modelCapabilities(dto.model_type ?? dto.model_types),
    status: stringValue(dto.status).trim(),
    maxTokens: numberValue(dto.max_tokens || dto.max_output),
  };
}

export function mapDefaultModel(value: unknown): PlatformDefaultModel | null {
  const dto = asRecord(value);
  const capability = stringValue(dto.model_type).trim();
  if (!capability) return null;
  return {
    capability,
    enabled: dto.enable === undefined ? true : booleanValue(dto.enable),
    instanceName: stringValue(dto.model_instance).trim(),
    modelId: stringValue(dto.model_id).trim(),
    modelName: stringValue(dto.model_name).trim(),
    providerName: stringValue(dto.model_provider).trim(),
  };
}

function mapTenantDefaultReference(
  capability: string,
  reference: string,
): PlatformDefaultModel | null {
  const parts = reference
    .split("@")
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length === 0) return null;

  const providerName = parts.length >= 2 ? (parts.pop() ?? "") : "";
  const instanceName = parts.length >= 2 ? (parts.pop() ?? "") : "default";
  const modelName = parts.join("@");
  if (!modelName) return null;

  return {
    capability,
    enabled: true,
    instanceName,
    modelId: "",
    modelName,
    providerName,
  };
}

/**
 * Combines the dedicated defaults endpoint with the composite selectors that
 * are returned by /users/me/models. The latter is required for the pinned
 * v0.26.4 runtime, which can persist a default while omitting it from
 * /models/default when historical inactive model rows also exist.
 */
export function mergePlatformDefaultModels(
  defaults: PlatformDefaultModel[],
  tenant: PlatformTenantModels,
): PlatformDefaultModel[] {
  const fallbackReferences: Array<[string, string]> = [
    ["chat", tenant.chatModelId],
    ["embedding", tenant.embeddingModelId],
    ["rerank", tenant.rerankModelId],
    ["asr", tenant.asrModelId],
    ["vision", tenant.imageToTextModelId],
    ["tts", tenant.textToSpeechModelId],
    ["ocr", tenant.ocrModelId],
  ];
  const seenCapabilities = new Set(defaults.map((item) => item.capability));
  const fallbacks = fallbackReferences.flatMap(([capability, reference]) => {
    if (!reference || seenCapabilities.has(capability)) return [];
    const mapped = mapTenantDefaultReference(capability, reference);
    return mapped ? [mapped] : [];
  });
  return [...defaults, ...fallbacks];
}

export function mapPipeline(value: unknown): PlatformPipeline | null {
  const dto = asRecord(value);
  const id = stringValue(dto.id).trim();
  if (!id) return null;
  return {
    id,
    title: stringValue(dto.title).trim() || id,
    description: stringValue(dto.description).trim(),
    filename: stringValue(dto.filename).trim(),
  };
}

export function mapProviderTask(value: unknown): PlatformProviderTask | null {
  const dto = asRecord(value);
  const id = stringValue(dto.task_id || dto.id).trim();
  if (!id) return null;
  return { id, status: stringValue(dto.status).trim() };
}

export function mapTaskSegment(value: unknown): PlatformTaskSegment | null {
  const dto = asRecord(value);
  const content = stringValue(dto.content);
  const index = numberValue(dto.index);
  if (index === null) return null;
  return { content, index };
}
