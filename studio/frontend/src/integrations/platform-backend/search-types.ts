import type { PlatformChatReference } from "./chat-completion-types";
import { asRecord, stringArray, stringValue } from "./model-types";

export interface PlatformSearchConfig {
  datasetIds: string[];
  documentIds: string[];
  chatModelId: string;
  similarityThreshold: number;
  vectorSimilarityWeight: number;
  topK: number;
  rerankId: string;
  useKnowledgeGraph: boolean;
  summary: boolean;
  highlight: boolean;
  keyword: boolean;
  webSearch: boolean;
  relatedSearch: boolean;
  queryMindMap: boolean;
  passthrough: Record<string, unknown>;
}

export interface PlatformSearchApp {
  id: string;
  tenantId: string;
  name: string;
  description: string;
  createdBy: string;
  ownerName: string;
  status: string;
  createTime: number | null;
  updateTime: number | null;
  hasConfig: boolean;
  config: PlatformSearchConfig;
}

export interface PlatformSearchListResult {
  items: PlatformSearchApp[];
  total: number;
}

export interface PlatformSearchStreamEvent {
  type: "answer" | "reference" | "done";
  answer?: string;
  reference?: PlatformChatReference;
}

const numberValue = (value: unknown, fallback: number) => {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};
const boolValue = (value: unknown, fallback = false) =>
  value == null ? fallback : value === true || value === 1 || value === "1";

export function mapPlatformSearchConfig(value: unknown): PlatformSearchConfig {
  const dto = asRecord(value);
  return {
    datasetIds: stringArray(dto.kb_ids),
    documentIds: stringArray(dto.doc_ids),
    chatModelId: stringValue(dto.chat_id).trim(),
    similarityThreshold: numberValue(dto.similarity_threshold, 0.2),
    vectorSimilarityWeight: numberValue(dto.vector_similarity_weight, 0.3),
    topK: Math.max(1, Math.round(numberValue(dto.top_k, 1024))),
    rerankId: stringValue(dto.rerank_id).trim(),
    useKnowledgeGraph: boolValue(dto.use_kg),
    summary: boolValue(dto.summary),
    highlight: boolValue(dto.highlight),
    keyword: boolValue(dto.keyword),
    webSearch: boolValue(dto.web_search),
    relatedSearch: boolValue(dto.related_search),
    queryMindMap: boolValue(dto.query_mindmap),
    passthrough: { ...dto },
  };
}

export function serializePlatformSearchConfig(
  config: PlatformSearchConfig,
): Record<string, unknown> {
  return {
    ...config.passthrough,
    kb_ids: config.datasetIds,
    doc_ids: config.documentIds,
    chat_id: config.chatModelId,
    similarity_threshold: config.similarityThreshold,
    vector_similarity_weight: config.vectorSimilarityWeight,
    top_k: config.topK,
    rerank_id: config.rerankId,
    use_kg: config.useKnowledgeGraph,
    summary: config.summary,
    highlight: config.highlight,
    keyword: config.keyword,
    web_search: config.webSearch,
    related_search: config.relatedSearch,
    query_mindmap: config.queryMindMap,
  };
}

export function mapPlatformSearchApp(value: unknown): PlatformSearchApp | null {
  const dto = asRecord(value);
  const id = stringValue(dto.id).trim();
  const name = stringValue(dto.name).trim();
  if (!id || !name) return null;
  return {
    id,
    tenantId: stringValue(dto.tenant_id).trim(),
    name,
    description: stringValue(dto.description),
    createdBy: stringValue(dto.created_by).trim(),
    ownerName: stringValue(dto.nickname).trim(),
    status: stringValue(dto.status),
    createTime:
      dto.create_time == null ? null : numberValue(dto.create_time, 0),
    updateTime:
      dto.update_time == null ? null : numberValue(dto.update_time, 0),
    hasConfig:
      Object.prototype.hasOwnProperty.call(dto, "search_config") &&
      dto.search_config != null,
    config: mapPlatformSearchConfig(dto.search_config),
  };
}
