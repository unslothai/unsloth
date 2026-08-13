import {
  createPlatformDataset,
  deletePlatformDatasets,
  getPlatformDataset,
  listPlatformDatasets,
  mapPipelineToDatasetFields,
  updatePlatformDataset,
  type PlatformDatasetChunkMethod,
  type PlatformDatasetDto,
  type PlatformDatasetPermission,
  type PlatformModel,
} from "@/integrations/platform-backend";

import type { KnowledgeBase } from "../types/rag";

export interface KnowledgeBaseListQuery {
  desc?: boolean;
  name?: string;
  orderBy?: "create_time" | "update_time";
  page: number;
  pageSize: number;
}

export interface KnowledgeBasePage {
  items: KnowledgeBase[];
  page: number;
  pageSize: number;
  total: number;
}

export interface KnowledgeBaseWriteInput {
  name: string;
  description?: string;
  embeddingModel: string;
  permission: PlatformDatasetPermission;
  chunkMethod?: PlatformDatasetChunkMethod;
  parserConfig?: Record<string, unknown>;
  pipelineId?: string;
}

const TENANT_MODEL_ID_PATTERN = /^[0-9a-f]{32}$/i;

/**
 * Dataset writes accept either the tenant-model primary key or the backend's
 * right-anchored model@instance@provider reference. Some legacy/custom model
 * rows expose a non-UUID model_id, so falling back to that opaque value makes
 * the Python dataset service parse a suffix in the model name as the provider.
 */
export function datasetEmbeddingModelReference(model: PlatformModel): string {
  const id = model.id.trim();
  if (TENANT_MODEL_ID_PATTERN.test(id)) return id;

  const name = model.name.trim();
  const provider = model.providerName.trim();
  const instance = model.instanceName.trim();
  if (name && provider) {
    return instance
      ? `${name}@${instance}@${provider}`
      : `${name}@${provider}`;
  }
  return id || name;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function dateValue(date: unknown, timestamp: unknown): string | null {
  if (typeof date === "string" && date.trim()) return date;
  const millis = typeof timestamp === "number" ? timestamp : Number(timestamp);
  return Number.isFinite(millis) && millis > 0
    ? new Date(millis).toISOString()
    : null;
}

function recordValue(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

export function mapDatasetToKnowledgeBase(
  dto: PlatformDatasetDto,
): KnowledgeBase {
  const id = stringValue(dto.id).trim();
  const name = stringValue(dto.name).trim();
  if (!id || !name) {
    throw new TypeError("Rag Platform dataset yanıtında id veya ad eksik.");
  }
  const count = Number(dto.document_count);
  const permission = stringValue(dto.permission);
  return {
    id,
    name,
    description:
      typeof dto.description === "string" ? dto.description : null,
    createdAt: dateValue(dto.create_date, dto.create_time),
    updatedAt: dateValue(dto.update_date, dto.update_time),
    documentCount: Number.isFinite(count) && count >= 0 ? count : 0,
    embeddingModel: stringValue(dto.embedding_model),
    permission: permission === "team" ? "team" : "me",
    chunkMethod: stringValue(dto.chunk_method || dto.parser_id) || "naive",
    parserConfig: recordValue(dto.parser_config),
    pipelineId: stringValue(dto.pipeline_id) || null,
  };
}

function toCreateRequest(input: KnowledgeBaseWriteInput) {
  const pipeline = mapPipelineToDatasetFields(input.pipelineId ?? "");
  return {
    name: input.name.trim(),
    ...(input.description?.trim()
      ? { description: input.description.trim() }
      : {}),
    embedding_model: input.embeddingModel.trim(),
    permission: input.permission,
    ...(pipeline ? pipeline : { chunk_method: input.chunkMethod ?? "naive" }),
    ...(input.parserConfig ? { parser_config: input.parserConfig } : {}),
  };
}

function toUpdateRequest(input: KnowledgeBaseWriteInput) {
  const pipeline = mapPipelineToDatasetFields(input.pipelineId ?? "");
  return {
    name: input.name.trim(),
    description: input.description?.trim() ?? "",
    embedding_model: input.embeddingModel.trim(),
    permission: input.permission,
    ...(pipeline
      ? pipeline
      : { parser_id: input.chunkMethod ?? "naive", parse_type: 1 as const }),
    ...(input.parserConfig ? { parser_config: input.parserConfig } : {}),
  };
}

export async function listKnowledgeBasePage(
  query: KnowledgeBaseListQuery,
  signal?: AbortSignal,
): Promise<KnowledgeBasePage> {
  const result = await listPlatformDatasets(
    {
      desc: query.desc,
      name: query.name,
      orderby: query.orderBy,
      page: query.page,
      pageSize: query.pageSize,
    },
    signal,
  );
  return {
    items: result.items.map(mapDatasetToKnowledgeBase),
    page: query.page,
    pageSize: query.pageSize,
    total: result.total,
  };
}

export async function listAllKnowledgeBases(
  signal?: AbortSignal,
): Promise<KnowledgeBase[]> {
  const pageSize = 100;
  const first = await listKnowledgeBasePage(
    { page: 1, pageSize, orderBy: "update_time", desc: true },
    signal,
  );
  const items = [...first.items];
  for (let page = 2; items.length < first.total; page += 1) {
    const next = await listKnowledgeBasePage(
      { page, pageSize, orderBy: "update_time", desc: true },
      signal,
    );
    items.push(...next.items);
    if (next.items.length === 0) break;
  }
  return items;
}

export async function getKnowledgeBase(
  id: string,
  signal?: AbortSignal,
): Promise<KnowledgeBase> {
  return mapDatasetToKnowledgeBase(await getPlatformDataset(id, signal));
}

export async function createDatasetKnowledgeBase(
  input: KnowledgeBaseWriteInput,
  signal?: AbortSignal,
): Promise<KnowledgeBase> {
  return mapDatasetToKnowledgeBase(
    await createPlatformDataset(toCreateRequest(input), signal),
  );
}

export async function updateDatasetKnowledgeBase(
  id: string,
  input: KnowledgeBaseWriteInput,
  signal?: AbortSignal,
): Promise<KnowledgeBase> {
  return mapDatasetToKnowledgeBase(
    await updatePlatformDataset(id, toUpdateRequest(input), signal),
  );
}

export function deleteDatasetKnowledgeBase(
  id: string,
  signal?: AbortSignal,
): Promise<void> {
  return deletePlatformDatasets([id], signal);
}
