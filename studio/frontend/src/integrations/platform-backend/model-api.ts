import { platformRequest } from "./client";
import { PlatformApiError } from "./errors";
import {
  type PlatformDefaultModel,
  type PlatformEmbeddingResult,
  type PlatformModel,
  type PlatformPipeline,
  type PlatformProvider,
  type PlatformProviderInstance,
  type PlatformProviderTask,
  type PlatformRerankResult,
  type PlatformTaskSegment,
  asRecord,
  mapDefaultModel,
  mapModel,
  mapPipeline,
  mapProvider,
  mapProviderInstance,
  mapProviderTask,
  mapTaskSegment,
  stringValue,
} from "./model-types";

const segment = (value: string) => encodeURIComponent(value.trim());
const list = <T>(value: unknown, mapper: (item: unknown) => T | null): T[] =>
  (Array.isArray(value) ? value : [])
    .map(mapper)
    .filter((item): item is T => item !== null);

const normalizedProviderBaseUrl = (providerName: string, value?: string) => {
  const baseUrl = value?.trim().replace(/\/+$/, "") ?? "";
  if (!baseUrl) return "";
  // The active hybrid runtime exposes custom OpenAI-compatible endpoints via
  // its VLLM driver. Mirror the backend's Python-side VLLM normalization so
  // models, chat and embedding suffixes resolve below the OpenAI `/v1` root.
  return providerName.trim().toLowerCase() === "vllm" &&
    !baseUrl.toLowerCase().endsWith("/v1")
    ? `${baseUrl}/v1`
    : baseUrl;
};

export interface ModelSelector {
  instanceName?: string;
  modelId?: string;
  modelName?: string;
  providerName?: string;
}

export interface ProviderInstanceInput {
  apiKey?: string;
  baseUrl?: string;
  instanceName: string;
  region?: string;
}

export interface ProviderModelInput {
  capabilities: string[];
  maxTokens?: number;
  modelName: string;
}

function selectorJson(selector: ModelSelector): Record<string, string> {
  return {
    ...(selector.providerName ? { provider_name: selector.providerName } : {}),
    ...(selector.instanceName ? { instance_name: selector.instanceName } : {}),
    ...(selector.modelName ? { model_name: selector.modelName } : {}),
    ...(selector.modelId ? { model_id: selector.modelId } : {}),
  };
}

export async function listAvailableProviders(
  signal?: AbortSignal,
): Promise<PlatformProvider[]> {
  const data = await platformRequest<unknown[]>("/providers", {
    query: { available: true },
    signal,
  });
  return list(data, mapProvider);
}

export async function listConfiguredProviders(
  signal?: AbortSignal,
): Promise<PlatformProvider[]> {
  const data = await platformRequest<unknown[]>("/providers", { signal });
  return list(data, mapProvider);
}

export async function getProvider(
  providerName: string,
  signal?: AbortSignal,
): Promise<PlatformProvider> {
  const data = await platformRequest<unknown>(
    `/providers/${segment(providerName)}`,
    { signal },
  );
  const mapped = mapProvider(data);
  if (!mapped)
    throw new PlatformApiError("Rag Platform provider yanıtı geçersiz.", {
      httpStatus: 200,
      code: "INVALID_RESPONSE",
      endpoint: "/providers/:provider",
    });
  return mapped;
}

export function addProvider(
  providerName: string,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest("/providers", {
    method: "PUT",
    json: { provider_name: providerName },
    signal,
  });
}

export function deleteProvider(
  providerName: string,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/providers/${segment(providerName)}`, {
    method: "DELETE",
    signal,
  });
}

export async function listProviderInstances(
  providerName: string,
  signal?: AbortSignal,
): Promise<PlatformProviderInstance[]> {
  const data = await platformRequest<unknown[]>(
    `/providers/${segment(providerName)}/instances`,
    { signal },
  );
  return list(data, mapProviderInstance);
}

export async function getProviderInstance(
  providerName: string,
  instanceName: string,
  signal?: AbortSignal,
): Promise<PlatformProviderInstance> {
  const data = await platformRequest<unknown>(
    `/providers/${segment(providerName)}/instances/${segment(instanceName)}`,
    { signal },
  );
  const mapped = mapProviderInstance(data);
  if (!mapped)
    throw new PlatformApiError(
      "Rag Platform provider instance yanıtı geçersiz.",
      {
        httpStatus: 200,
        code: "INVALID_RESPONSE",
        endpoint: "/providers/:provider/instances/:instance",
      },
    );
  return mapped;
}

export function createProviderInstance(
  providerName: string,
  input: ProviderInstanceInput,
  signal?: AbortSignal,
): Promise<unknown> {
  const baseUrl = normalizedProviderBaseUrl(providerName, input.baseUrl);
  return platformRequest(`/providers/${segment(providerName)}/instances`, {
    method: "POST",
    json: {
      instance_name: input.instanceName.trim(),
      ...(input.apiKey ? { api_key: input.apiKey } : {}),
      ...(baseUrl ? { base_url: baseUrl } : {}),
      ...(input.region ? { region: input.region.trim() } : {}),
    },
    signal,
  });
}

export function updateProviderInstance(
  providerName: string,
  currentName: string,
  input: ProviderInstanceInput & { verify?: boolean },
  signal?: AbortSignal,
): Promise<unknown> {
  const baseUrl = normalizedProviderBaseUrl(providerName, input.baseUrl);
  return platformRequest(
    `/providers/${segment(providerName)}/instances/${segment(currentName)}`,
    {
      method: "PUT",
      json: {
        instance_name: input.instanceName.trim(),
        ...(input.apiKey ? { api_key: input.apiKey } : {}),
        base_url: baseUrl,
        region: input.region?.trim() ?? "",
        verify: input.verify ?? true,
      },
      signal,
    },
  );
}

export function deleteProviderInstances(
  providerName: string,
  instanceNames: string[],
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/providers/${segment(providerName)}/instances`, {
    method: "DELETE",
    json: { instances: instanceNames },
    signal,
  });
}

export function testProviderInstanceConnection(
  providerName: string,
  instanceName: string,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(
    `/providers/${segment(providerName)}/instances/${segment(instanceName)}/connection`,
    { signal, getRetries: 0 },
  );
}

export function testProviderConnection(
  providerName: string,
  input: Omit<ProviderInstanceInput, "instanceName"> & { instanceId?: string },
  signal?: AbortSignal,
): Promise<unknown> {
  const baseUrl = normalizedProviderBaseUrl(providerName, input.baseUrl);
  return platformRequest(`/providers/${segment(providerName)}/connection`, {
    method: "POST",
    json: {
      ...(input.apiKey ? { api_key: input.apiKey } : {}),
      ...(baseUrl ? { base_url: baseUrl } : {}),
      ...(input.region ? { region: input.region.trim() } : {}),
      ...(input.instanceId ? { instance_id: input.instanceId } : {}),
      model_info: [],
    },
    signal,
  });
}

export function getProviderInstanceBalance(
  providerName: string,
  instanceName: string,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(
    `/providers/${segment(providerName)}/instances/${segment(instanceName)}/balance`,
    { signal, getRetries: 0 },
  );
}

export async function listProviderTasks(
  providerName: string,
  instanceName: string,
  signal?: AbortSignal,
): Promise<PlatformProviderTask[]> {
  const data = await platformRequest<unknown[]>(
    `/providers/${segment(providerName)}/instances/${segment(instanceName)}/tasks`,
    { signal },
  );
  return list(data, mapProviderTask);
}

export async function getProviderTask(
  providerName: string,
  instanceName: string,
  taskId: string,
  signal?: AbortSignal,
): Promise<PlatformTaskSegment[]> {
  const data = await platformRequest<unknown>(
    `/providers/${segment(providerName)}/instances/${segment(instanceName)}/tasks/${segment(taskId)}`,
    { signal },
  );
  return list(asRecord(data).segments, mapTaskSegment);
}

export async function listTenantModels(
  signal?: AbortSignal,
): Promise<PlatformModel[]> {
  const data = await platformRequest<unknown[]>("/models", { signal });
  return list(data, mapModel);
}

export async function listProviderModels(
  providerName: string,
  signal?: AbortSignal,
): Promise<PlatformModel[]> {
  const data = await platformRequest<unknown[]>(
    `/providers/${segment(providerName)}/models`,
    { signal },
  );
  return list(data, mapModel);
}

export async function getProviderModel(
  providerName: string,
  modelName: string,
  signal?: AbortSignal,
): Promise<PlatformModel> {
  const data = await platformRequest<unknown>(
    `/providers/${segment(providerName)}/models/${segment(modelName)}`,
    { signal },
  );
  const mapped = mapModel(data);
  if (!mapped)
    throw new PlatformApiError("Rag Platform model yanıtı geçersiz.", {
      httpStatus: 200,
      code: "INVALID_RESPONSE",
      endpoint: "/providers/:provider/models/:model",
    });
  return mapped;
}

export async function listInstanceModels(
  providerName: string,
  instanceName: string,
  signal?: AbortSignal,
): Promise<PlatformModel[]> {
  const data = await platformRequest<unknown[]>(
    `/providers/${segment(providerName)}/instances/${segment(instanceName)}/models`,
    { signal },
  );
  return list(data, mapModel).map((model) => ({
    ...model,
    providerName: model.providerName || providerName,
    instanceName: model.instanceName || instanceName,
  }));
}

/**
 * Ask the provider itself for the models reachable through a saved instance.
 *
 * This is intentionally separate from `listInstanceModels`: the unqualified
 * route only returns models already persisted in Rag Platform, while
 * `supported=true` uses the server-held credential to discover the remote
 * catalog without exposing that credential back to the browser.
 */
export async function listSupportedInstanceModels(
  providerName: string,
  instanceName: string,
  signal?: AbortSignal,
): Promise<PlatformModel[]> {
  const data = await platformRequest<unknown[]>(
    `/providers/${segment(providerName)}/instances/${segment(instanceName)}/models`,
    { query: { supported: true }, signal },
  );
  return list(data, mapModel).map((model) => ({
    ...model,
    providerName: model.providerName || providerName,
    instanceName: model.instanceName || instanceName,
  }));
}

export function addInstanceModel(
  providerName: string,
  instanceName: string,
  input: ProviderModelInput,
  signal?: AbortSignal,
): Promise<unknown> {
  const modelName = input.modelName.trim();
  const capabilities = input.capabilities
    .map((capability) => capability.trim())
    .filter(Boolean);
  const maxTokens = input.maxTokens ?? 8192;
  return platformRequest(
    `/providers/${segment(providerName)}/instances/${segment(instanceName)}/models`,
    {
      method: "POST",
      json: {
        // Backend main accepts the canonical single-model fields. The pinned
        // v0.26.4 Go runtime accepts the nested `models` batch shape. Sending
        // both is forward/backward compatible because each decoder ignores
        // the fields belonging to the other contract.
        model_name: modelName,
        model_type: capabilities,
        max_tokens: maxTokens,
        extra: {},
        models: [
          {
            model_name: modelName,
            model_types: capabilities,
            max_tokens: maxTokens,
            max_dimension: 0,
            dimensions: [],
          },
        ],
      },
      signal,
    },
  );
}

export function updateInstanceModel(
  providerName: string,
  instanceName: string,
  modelName: string,
  update: {
    capabilities?: string[];
    maxTokens?: number;
    status?: "active" | "inactive";
  },
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(
    `/providers/${segment(providerName)}/instances/${segment(instanceName)}/models/${segment(modelName)}`,
    {
      method: "PATCH",
      json: {
        ...(update.capabilities ? { model_type: update.capabilities } : {}),
        ...(update.maxTokens ? { max_tokens: update.maxTokens } : {}),
        ...(update.status ? { status: update.status } : {}),
      },
      signal,
    },
  );
}

export function deleteInstanceModels(
  providerName: string,
  instanceName: string,
  modelNames: string[],
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(
    `/providers/${segment(providerName)}/instances/${segment(instanceName)}/models`,
    {
      method: "DELETE",
      // Same mixed-runtime compatibility rule as model creation: backend main
      // reads `model_name`, while v0.26.4 reads `models`.
      json: { model_name: modelNames, models: modelNames },
      signal,
    },
  );
}

export async function getDefaultModels(
  signal?: AbortSignal,
): Promise<PlatformDefaultModel[]> {
  const data = await platformRequest<unknown>("/models/default", { signal });
  return list(asRecord(data).models, mapDefaultModel);
}

export function setDefaultModel(
  model: PlatformModel,
  capability: string,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest("/models/default", {
    method: "PATCH",
    json: {
      model_provider: model.providerName,
      model_instance: model.instanceName,
      model_name: model.name,
      model_id: model.id,
      model_type: capability,
    },
    signal,
  });
}

export interface ChatToModelResult {
  answer: string;
  reasoning: string;
  usage: unknown;
}
export async function chatToModel(
  selector: ModelSelector,
  messages: Array<{ role: string; content: string }>,
  signal?: AbortSignal,
): Promise<ChatToModelResult> {
  const endpoint = "/chat/to_model";
  const raw = await platformRequest<unknown>(endpoint, {
    method: "POST",
    json: {
      ...selectorJson(selector),
      messages,
      stream: false,
      thinking: false,
    },
    responseType: "json",
    signal,
    timeoutMs: 60_000,
  });
  const dto = asRecord(raw);
  if (dto.code !== 0)
    throw new PlatformApiError(
      stringValue(dto.message) || "Rag Platform model isteğini reddetti.",
      {
        httpStatus: 200,
        code: typeof dto.code === "number" ? dto.code : "INVALID_RESPONSE",
        endpoint,
      },
    );
  return {
    answer: stringValue(dto.answer),
    reasoning: stringValue(dto.reasoning_content),
    usage: dto.usage,
  };
}

export async function createEmbeddings(
  selector: ModelSelector,
  texts: string[],
  dimension: number,
  signal?: AbortSignal,
): Promise<PlatformEmbeddingResult[]> {
  const data = await platformRequest<unknown[]>("/embeddings", {
    method: "POST",
    json: { ...selectorJson(selector), texts, dimension },
    signal,
    timeoutMs: 60_000,
  });
  return (Array.isArray(data) ? data : []).map((item) => {
    const dto = asRecord(item);
    return {
      index: Number(dto.index) || 0,
      tokenCount: Number(dto.token_count) || 0,
      vector: Array.isArray(dto.embedding)
        ? dto.embedding.filter(
            (value): value is number => typeof value === "number",
          )
        : [],
    };
  });
}

export async function rerankDocuments(
  selector: ModelSelector,
  query: string,
  documents: string[],
  topN: number,
  signal?: AbortSignal,
): Promise<PlatformRerankResult[]> {
  const data = await platformRequest<unknown[]>("/rerank", {
    method: "POST",
    json: { ...selectorJson(selector), query, documents, top_n: topN },
    signal,
    timeoutMs: 60_000,
  });
  return (Array.isArray(data) ? data : []).map((item) => {
    const dto = asRecord(item);
    return {
      index: Number(dto.index) || 0,
      relevanceScore: Number(dto.relevance_score) || 0,
    };
  });
}

export async function transcribeAudio(
  selector: ModelSelector,
  fileBase64: string,
  languages: string[],
  signal?: AbortSignal,
): Promise<string> {
  const data = await platformRequest<unknown>("/audio/transcriptions", {
    method: "POST",
    json: {
      ...selectorJson(selector),
      file: fileBase64,
      language: languages,
      prompt: 0,
      stream: false,
      asr_config: {},
    },
    signal,
    timeoutMs: 120_000,
  });
  return stringValue(asRecord(data).text);
}

export async function synthesizeSpeech(
  selector: ModelSelector,
  value: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const data = await platformRequest<unknown>("/audio/speech", {
    method: "POST",
    json: {
      ...selectorJson(selector),
      text: value,
      stream: false,
      tts_config: {},
    },
    signal,
    timeoutMs: 120_000,
  });
  const encoded = stringValue(asRecord(data).audio);
  const bytes = Uint8Array.from(atob(encoded), (character) =>
    character.charCodeAt(0),
  );
  return new Blob([bytes], { type: "audio/mpeg" });
}

export async function ocrFile(
  selector: ModelSelector,
  contentBase64: string,
  signal?: AbortSignal,
): Promise<string> {
  const data = await platformRequest<unknown>("/file/ocr", {
    method: "POST",
    json: { ...selectorJson(selector), content: contentBase64 },
    signal,
    timeoutMs: 120_000,
  });
  return stringValue(asRecord(data).text);
}

export async function parseFile(
  selector: ModelSelector,
  contentBase64: string,
  signal?: AbortSignal,
): Promise<string> {
  const data = await platformRequest<unknown>("/file/parse", {
    method: "POST",
    json: { ...selectorJson(selector), content: contentBase64 },
    signal,
    timeoutMs: 120_000,
  });
  return stringValue(asRecord(data).task_id);
}

export async function listPipelines(
  signal?: AbortSignal,
): Promise<PlatformPipeline[]> {
  const data = await platformRequest<unknown>("/pipelines", {
    token: null,
    query: { type: "builtin" },
    signal,
    getRetries: 0,
  });
  return list(asRecord(data).canvas, mapPipeline);
}

export async function getPipelineDsl(
  id: string,
  signal?: AbortSignal,
): Promise<unknown> {
  const data = await platformRequest<unknown>(`/pipelines/${segment(id)}`, {
    token: null,
    signal,
    getRetries: 0,
  });
  return asRecord(data).dsl;
}
