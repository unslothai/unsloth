import type { PlatformDocument } from "./document-types";

export interface PlatformChunkDto {
  id?: unknown;
  chunk_id?: unknown;
  dataset_id?: unknown;
  kb_id?: unknown;
  document_id?: unknown;
  doc_id?: unknown;
  document_name?: unknown;
  docnm_kwd?: unknown;
  content?: unknown;
  content_with_weight?: unknown;
  important_keywords?: unknown;
  important_kwd?: unknown;
  questions?: unknown;
  question_kwd?: unknown;
  available?: unknown;
  available_int?: unknown;
  positions?: unknown;
  position_int?: unknown;
  image_id?: unknown;
  img_id?: unknown;
  similarity?: unknown;
  score?: unknown;
  vector_similarity?: unknown;
  term_similarity?: unknown;
  rerank_score?: unknown;
  doc_type?: unknown;
  doc_type_kwd?: unknown;
  [key: string]: unknown;
}

export interface PlatformChunkScore {
  similarity: number | null;
  vectorSimilarity: number | null;
  termSimilarity: number | null;
  rerankScore: number | null;
}

export interface PlatformChunk {
  id: string;
  datasetId: string;
  documentId: string;
  documentName: string;
  content: string;
  importantKeywords: string[];
  questions: string[];
  enabled: boolean;
  positions: unknown[];
  pageNumber: number | null;
  imageId: string | null;
  documentType: string;
  normalizedScore: number | null;
  scores: PlatformChunkScore;
}

export interface PlatformChunkListResult {
  items: PlatformChunk[];
  total: number;
  document: Record<string, unknown> | null;
}

export interface PlatformChunkListOptions {
  page?: number;
  pageSize?: number;
  keywords?: string;
  available?: boolean;
}

export interface PlatformChunkDraft {
  content: string;
  importantKeywords?: string[];
  questions?: string[];
  enabled?: boolean;
}

export interface PlatformRetrievalRequest {
  datasetIds: string[];
  question: string;
  documentIds?: string[];
  page?: number;
  pageSize?: number;
  topK?: number;
  similarityThreshold?: number;
  vectorSimilarityWeight?: number;
  rerankId?: string;
  highlight?: boolean;
}

export interface PlatformRetrievalResult {
  items: PlatformChunk[];
  total: number;
  documentAggregations: Array<Record<string, unknown>>;
}

export interface PlatformStructureEntity {
  id: string;
  name: string;
  description: string | null;
  sourceChunkIds: string[];
}

export interface PlatformStructureRelation {
  id: string;
  source: string;
  target: string;
  description: string | null;
}

export interface PlatformStructureTemplate {
  id: string;
  name: string;
  kind: string;
  entities: PlatformStructureEntity[];
  relations: PlatformStructureRelation[];
}

export interface PlatformStructureGraph {
  templates: PlatformStructureTemplate[];
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function numberValue(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizedScore(value: number | null): number | null {
  if (value === null) return null;
  const ratio = value > 1 && value <= 100 ? value / 100 : value;
  return Math.max(0, Math.min(1, ratio));
}

function stringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map(stringValue)
      .filter((entry, index, rows) => entry && rows.indexOf(entry) === index);
  }
  if (typeof value !== "string") return [];
  return value
    .split(value.includes("###") ? "###" : ",")
    .map((entry) => entry.trim())
    .filter((entry, index, rows) => entry && rows.indexOf(entry) === index);
}

function enabledValue(dto: PlatformChunkDto): boolean {
  if (typeof dto.available === "boolean") return dto.available;
  const raw = dto.available_int ?? dto.available;
  if (raw === 0 || raw === "0" || raw === false) return false;
  return true;
}

function positionArray(dto: PlatformChunkDto): unknown[] {
  const raw = dto.positions ?? dto.position_int;
  if (Array.isArray(raw)) return raw;
  return raw == null || raw === "" ? [] : [raw];
}

function firstPositionNumber(value: unknown): number | null {
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = firstPositionNumber(item);
      if (found !== null) return found;
    }
    return null;
  }
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const match = value.match(/^-?\d+/);
    if (match) return Number(match[0]);
  }
  return null;
}

export function mapPlatformChunk(
  dto: PlatformChunkDto,
  fallbacks: { datasetId?: string; documentId?: string } = {},
): PlatformChunk {
  const id = stringValue(dto.chunk_id ?? dto.id);
  if (!id) throw new TypeError("Rag Platform chunk yanıtında id eksik.");
  const positions = positionArray(dto);
  const rawPage = firstPositionNumber(positions);
  const similarity = numberValue(dto.similarity ?? dto.score);
  const vectorSimilarity = numberValue(dto.vector_similarity);
  const termSimilarity = numberValue(dto.term_similarity);
  const rerankScore = numberValue(dto.rerank_score);
  const score = normalizedScore(
    rerankScore ?? similarity ?? vectorSimilarity ?? termSimilarity,
  );
  return {
    id,
    datasetId:
      stringValue(dto.dataset_id ?? dto.kb_id) || fallbacks.datasetId || "",
    documentId:
      stringValue(dto.document_id ?? dto.doc_id) || fallbacks.documentId || "",
    documentName: stringValue(dto.document_name ?? dto.docnm_kwd),
    content: stringValue(dto.content ?? dto.content_with_weight),
    importantKeywords: stringList(dto.important_keywords ?? dto.important_kwd),
    questions: stringList(dto.questions ?? dto.question_kwd),
    enabled: enabledValue(dto),
    positions,
    pageNumber: rawPage === null ? null : Math.max(1, Math.trunc(rawPage || 1)),
    imageId: stringValue(dto.image_id ?? dto.img_id) || null,
    documentType: stringValue(dto.doc_type ?? dto.doc_type_kwd),
    normalizedScore: score,
    scores: {
      similarity: normalizedScore(similarity),
      vectorSimilarity: normalizedScore(vectorSimilarity),
      termSimilarity: normalizedScore(termSimilarity),
      rerankScore: normalizedScore(rerankScore),
    },
  };
}

function objectRows(value: unknown): PlatformChunkDto[] {
  return Array.isArray(value)
    ? (value.filter(
        (entry) => typeof entry === "object" && entry !== null,
      ) as PlatformChunkDto[])
    : [];
}

export function mapPlatformChunkList(
  value: unknown,
  fallbacks: { datasetId?: string; documentId?: string } = {},
): PlatformChunkListResult {
  const data = typeof value === "object" && value !== null ? value : {};
  const rows = objectRows("chunks" in data ? data.chunks : []);
  const total = numberValue("total" in data ? data.total : null);
  return {
    items: rows.map((row) => mapPlatformChunk(row, fallbacks)),
    total: Math.max(0, Math.trunc(total ?? rows.length)),
    document:
      "doc" in data && typeof data.doc === "object" && data.doc !== null
        ? (data.doc as Record<string, unknown>)
        : null,
  };
}

export function mapPlatformRetrieval(
  value: unknown,
  fallbackDatasetId = "",
): PlatformRetrievalResult {
  const data = typeof value === "object" && value !== null ? value : {};
  const rows = objectRows("chunks" in data ? data.chunks : []);
  const aggregations =
    "doc_aggs" in data && Array.isArray(data.doc_aggs)
      ? data.doc_aggs.filter(
          (entry): entry is Record<string, unknown> =>
            typeof entry === "object" && entry !== null,
        )
      : [];
  const total = numberValue("total" in data ? data.total : null);
  return {
    items: rows.map((row) =>
      mapPlatformChunk(row, { datasetId: fallbackDatasetId }),
    ),
    total: Math.max(0, Math.trunc(total ?? rows.length)),
    documentAggregations: aggregations,
  };
}

function objectValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

function entityName(row: Record<string, unknown>): string {
  return (
    stringValue(row.name) ||
    stringValue(row.entity_name) ||
    stringValue(row.label) ||
    stringValue(row.id) ||
    "İsimsiz varlık"
  );
}

export function mapPlatformStructureGraph(
  value: unknown,
): PlatformStructureGraph {
  const data = objectValue(value);
  const templates = Array.isArray(data.templates) ? data.templates : [];
  return {
    templates: templates
      .map(objectValue)
      .map((template, templateIndex): PlatformStructureTemplate => {
        const entities = Array.isArray(template.entities)
          ? template.entities.map(objectValue)
          : [];
        const relations = Array.isArray(template.relations)
          ? template.relations.map(objectValue)
          : [];
        return {
          id:
            stringValue(template.template_id) ||
            `template-${templateIndex + 1}`,
          name:
            stringValue(template.template_name) ||
            stringValue(template.template_id) ||
            `Yapı ${templateIndex + 1}`,
          kind: stringValue(template.kind) || "unknown",
          entities: entities.map((entity, entityIndex) => ({
            id:
              stringValue(entity.id ?? entity.entity_id) ||
              `entity-${entityIndex + 1}`,
            name: entityName(entity),
            description:
              stringValue(
                entity.description ?? entity.content ?? entity.summary,
              ) || null,
            sourceChunkIds: stringList(
              entity.source_chunk_ids ?? entity.chunk_ids,
            ),
          })),
          relations: relations.map((relation, relationIndex) => ({
            id: stringValue(relation.id) || `relation-${relationIndex + 1}`,
            source: stringValue(
              relation.source ?? relation.source_id ?? relation.from,
            ),
            target: stringValue(
              relation.target ?? relation.target_id ?? relation.to,
            ),
            description:
              stringValue(
                relation.description ?? relation.content ?? relation.label,
              ) || null,
          })),
        };
      }),
  };
}

export function chunkPreviewDocument(
  chunk: PlatformChunk,
  fallbackDatasetId: string,
): PlatformDocument {
  return {
    id: chunk.documentId,
    datasetId: chunk.datasetId || fallbackDatasetId,
    name: chunk.documentName || "Kaynak belge",
    thumbnail: null,
    sizeBytes: 0,
    sourceType: chunk.documentType || "unknown",
    location: null,
    tokenCount: 0,
    chunkCount: 0,
    progress: 1,
    progressMessage: null,
    processDuration: 0,
    suffix: chunk.documentType.replace(/^\./, ""),
    run: "DONE",
    backendStatus: "1",
    status: "completed",
    parserId: "",
    chunkMethod: "",
    pipelineId: null,
    pipelineName: null,
    createdAt: null,
    updatedAt: null,
  };
}
