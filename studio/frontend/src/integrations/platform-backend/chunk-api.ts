import { platformRequest } from "./client";
import {
  mapPlatformChunk,
  mapPlatformChunkList,
  mapPlatformRetrieval,
  mapPlatformStructureGraph,
  type PlatformChunk,
  type PlatformChunkDraft,
  type PlatformChunkDto,
  type PlatformChunkListOptions,
  type PlatformChunkListResult,
  type PlatformRetrievalRequest,
  type PlatformRetrievalResult,
  type PlatformStructureGraph,
} from "./chunk-types";

function encode(value: string): string {
  return encodeURIComponent(value);
}

function listPayload(
  datasetId: string,
  documentId: string,
  options: PlatformChunkListOptions,
) {
  const page = Math.max(1, Math.trunc(options.page ?? 1));
  const pageSize = Math.min(
    200,
    Math.max(1, Math.trunc(options.pageSize ?? 50)),
  );
  return {
    page,
    pageSize,
    keywords: options.keywords?.trim() || undefined,
    available: options.available,
    body: {
      dataset_id: datasetId,
      doc_id: documentId,
      page,
      size: pageSize,
      ...(options.keywords?.trim()
        ? { keywords: options.keywords.trim() }
        : {}),
      ...(options.available === undefined
        ? {}
        : { available_int: options.available ? 1 : 0 }),
    },
  };
}

export async function listDocumentChunks(
  datasetId: string,
  documentId: string,
  options: PlatformChunkListOptions = {},
  signal?: AbortSignal,
): Promise<PlatformChunkListResult> {
  const payload = listPayload(datasetId, documentId, options);
  const value = await platformRequest<unknown>(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}/chunks`,
    {
      query: {
        page: payload.page,
        page_size: payload.pageSize,
        ...(payload.keywords ? { keywords: payload.keywords } : {}),
        ...(payload.available === undefined
          ? {}
          : { available: payload.available }),
      },
      signal,
    },
  );
  return mapPlatformChunkList(value, { datasetId, documentId });
}

/**
 * Protocol-compatibility contract for the v0.26.4 Go POST-with-body listing.
 * Product UI uses the canonical dataset/document GET route above.
 */
export async function listDocumentChunksCompatibility(
  datasetId: string,
  documentId: string,
  options: PlatformChunkListOptions = {},
  signal?: AbortSignal,
): Promise<PlatformChunkListResult> {
  const payload = listPayload(datasetId, documentId, options);
  const value = await platformRequest<unknown>("/chunk/list", {
    method: "POST",
    json: payload.body,
    signal,
  });
  return mapPlatformChunkList(value, { datasetId, documentId });
}

export async function getDocumentChunk(
  datasetId: string,
  documentId: string,
  chunkId: string,
  signal?: AbortSignal,
): Promise<PlatformChunk> {
  const value = await platformRequest<PlatformChunkDto>(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}/chunks/${encode(chunkId)}`,
    { signal },
  );
  return mapPlatformChunk(value, { datasetId, documentId });
}

export async function createDocumentChunk(
  datasetId: string,
  documentId: string,
  draft: PlatformChunkDraft,
  signal?: AbortSignal,
): Promise<PlatformChunk> {
  const value = await platformRequest<unknown>(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}/chunks`,
    {
      method: "POST",
      json: {
        content: draft.content,
        important_keywords: draft.importantKeywords ?? [],
        questions: draft.questions ?? [],
      },
      signal,
    },
  );
  const data =
    typeof value === "object" && value !== null && "chunk" in value
      ? value.chunk
      : value;
  return mapPlatformChunk(data as PlatformChunkDto, {
    datasetId,
    documentId,
  });
}

export function updateDocumentChunk(
  datasetId: string,
  documentId: string,
  chunkId: string,
  draft: Partial<PlatformChunkDraft>,
  signal?: AbortSignal,
): Promise<void> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}/chunks/${encode(chunkId)}`,
    {
      method: "PATCH",
      json: {
        ...(draft.content === undefined ? {} : { content: draft.content }),
        ...(draft.importantKeywords === undefined
          ? {}
          : { important_keywords: draft.importantKeywords }),
        ...(draft.questions === undefined
          ? {}
          : { questions: draft.questions }),
        ...(draft.enabled === undefined ? {} : { available: draft.enabled }),
      },
      signal,
    },
  );
}

/** Deprecated protocol alias. No product UI calls this PUT route. */
export function updateDocumentChunkCompatibility(
  datasetId: string,
  documentId: string,
  chunkId: string,
  draft: Partial<PlatformChunkDraft>,
  signal?: AbortSignal,
): Promise<void> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}/chunks/${encode(chunkId)}`,
    {
      method: "PUT",
      json: {
        ...(draft.content === undefined ? {} : { content: draft.content }),
        ...(draft.importantKeywords === undefined
          ? {}
          : { important_keywords: draft.importantKeywords }),
        ...(draft.questions === undefined
          ? {}
          : { questions: draft.questions }),
        ...(draft.enabled === undefined ? {} : { available: draft.enabled }),
      },
      signal,
    },
  );
}

export function setDocumentChunksEnabled(
  datasetId: string,
  documentId: string,
  chunkIds: string[],
  enabled: boolean,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}/chunks`,
    {
      method: "PATCH",
      json: { chunk_ids: chunkIds, available: enabled },
      signal,
    },
  );
}

export function deleteDocumentChunks(
  datasetId: string,
  documentId: string,
  chunkIds: string[],
  signal?: AbortSignal,
): Promise<void> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}/chunks`,
    {
      method: "DELETE",
      json: { chunk_ids: chunkIds },
      signal,
    },
  );
}

/** Legacy parse/stop aliases are contract-tested but not product UI routes. */
export function parseDatasetChunksCompatibility(
  datasetId: string,
  documentIds: string[],
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/datasets/${encode(datasetId)}/chunks`, {
    method: "POST",
    json: { document_ids: documentIds },
    signal,
  });
}

export function stopDatasetChunksCompatibility(
  datasetId: string,
  documentIds: string[],
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/datasets/${encode(datasetId)}/chunks`, {
    method: "DELETE",
    json: { document_ids: documentIds },
    signal,
  });
}

export async function retrievePlatformChunks(
  request: PlatformRetrievalRequest,
  signal?: AbortSignal,
): Promise<PlatformRetrievalResult> {
  const question = request.question.trim();
  const datasetIds = request.datasetIds.filter(Boolean);
  const value = await platformRequest<unknown>("/retrieval", {
    method: "POST",
    json: {
      question,
      dataset_ids: datasetIds,
      document_ids: request.documentIds?.filter(Boolean) ?? [],
      page: Math.max(1, Math.trunc(request.page ?? 1)),
      page_size: Math.min(
        100,
        Math.max(1, Math.trunc(request.pageSize ?? request.topK ?? 10)),
      ),
      top_k: Math.min(1024, Math.max(1, Math.trunc(request.topK ?? 10))),
      similarity_threshold: Math.max(
        0,
        Math.min(1, request.similarityThreshold ?? 0.2),
      ),
      vector_similarity_weight: Math.max(
        0,
        Math.min(1, request.vectorSimilarityWeight ?? 0.3),
      ),
      highlight: request.highlight ?? true,
      ...(request.rerankId?.trim()
        ? { rerank_id: request.rerankId.trim() }
        : {}),
    },
    signal,
    timeoutMs: 60_000,
  });
  return mapPlatformRetrieval(value, datasetIds[0] ?? "");
}

/** Dataset-scoped Go search is a compatibility contract; UI uses /retrieval. */
export async function searchDatasetChunksCompatibility(
  datasetId: string,
  request: Omit<PlatformRetrievalRequest, "datasetIds">,
  signal?: AbortSignal,
): Promise<PlatformRetrievalResult> {
  const value = await platformRequest<unknown>(
    `/datasets/${encode(datasetId)}/search`,
    {
      method: "POST",
      json: {
        question: request.question.trim(),
        page: Math.max(1, Math.trunc(request.page ?? 1)),
        size: Math.min(
          100,
          Math.max(1, Math.trunc(request.pageSize ?? request.topK ?? 10)),
        ),
        doc_ids: request.documentIds?.filter(Boolean) ?? [],
        top_k: Math.min(1024, Math.max(1, Math.trunc(request.topK ?? 10))),
        similarity_threshold: Math.max(
          0,
          Math.min(1, request.similarityThreshold ?? 0.2),
        ),
        vector_similarity_weight: Math.max(
          0,
          Math.min(1, request.vectorSimilarityWeight ?? 0.3),
        ),
        ...(request.rerankId?.trim()
          ? { rerank_id: request.rerankId.trim() }
          : {}),
      },
      signal,
      timeoutMs: 60_000,
    },
  );
  return mapPlatformRetrieval(value, datasetId);
}

export async function getDocumentStructureGraph(
  datasetId: string,
  documentId: string,
  keywords = "",
  signal?: AbortSignal,
): Promise<PlatformStructureGraph> {
  const value = await platformRequest<unknown>(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}/structure/graph`,
    {
      query: keywords.trim() ? { keywords: keywords.trim() } : undefined,
      signal,
      timeoutMs: 45_000,
    },
  );
  return mapPlatformStructureGraph(value);
}

export async function deleteDocumentStructureGraph(
  datasetId: string,
  documentId: string,
  templateId: string,
  signal?: AbortSignal,
): Promise<number> {
  const value = await platformRequest<unknown>(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}/structure/graph`,
    {
      method: "DELETE",
      json: { template_id: templateId },
      signal,
    },
  );
  if (typeof value === "object" && value !== null && "deleted" in value) {
    const deleted = Number(value.deleted);
    return Number.isFinite(deleted) ? Math.max(0, Math.trunc(deleted)) : 0;
  }
  return 0;
}
