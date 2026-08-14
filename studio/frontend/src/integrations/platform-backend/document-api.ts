import { platformRequest } from "./client";
import { unwrapPlatformEnvelope } from "./envelope";
import { PlatformApiError } from "./errors";
import {
  mapPlatformDocument,
  mapPlatformIngestionTask,
  type PlatformAsset,
  type PlatformDatasetDocumentDto,
  type PlatformDatasetDocumentListResult,
  type PlatformDocumentUploadResult,
  type PlatformGenericDocumentDto,
  type PlatformIngestionTask,
  type PlatformIngestionTaskDto,
  type PlatformUploadInspection,
  PLATFORM_DOCUMENT_MAX_BYTES,
} from "./document-types";
import type { PlatformEnvelope } from "./types";

interface DocumentListData {
  docs?: unknown;
  total?: unknown;
}

export interface PlatformDocumentListOptions {
  page?: number;
  pageSize?: number;
  keywords?: string;
}

function encode(value: string): string {
  return encodeURIComponent(value);
}

function finiteNonNegative(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function asDocumentArray(value: unknown): PlatformDatasetDocumentDto[] {
  return Array.isArray(value)
    ? (value.filter(
        (entry) => typeof entry === "object" && entry !== null,
      ) as PlatformDatasetDocumentDto[])
    : [];
}

export async function listDatasetDocuments(
  datasetId: string,
  options: PlatformDocumentListOptions = {},
  signal?: AbortSignal,
): Promise<PlatformDatasetDocumentListResult> {
  const page = Math.max(1, Math.trunc(options.page ?? 1));
  const pageSize = Math.min(100, Math.max(1, Math.trunc(options.pageSize ?? 10)));
  const keywords = options.keywords?.trim();
  const data = await platformRequest<DocumentListData>(
    `/datasets/${encode(datasetId)}/documents`,
    {
      query: {
        page,
        page_size: pageSize,
        orderby: "update_time",
        desc: true,
        ...(keywords ? { keywords } : {}),
      },
      signal,
    },
  );
  const rows = asDocumentArray(data?.docs).map((dto) =>
    mapPlatformDocument(dto, datasetId),
  );
  return { items: rows, total: finiteNonNegative(data?.total, rows.length) };
}

export async function uploadDatasetDocuments(
  datasetId: string,
  files: File[],
  signal?: AbortSignal,
): Promise<PlatformDocumentUploadResult> {
  const endpoint = `/datasets/${encode(datasetId)}/documents`;
  const form = new FormData();
  for (const file of files) form.append("file", file, file.name);
  const envelope = await platformRequest<
    PlatformEnvelope<PlatformDatasetDocumentDto[]>
  >(endpoint, {
    method: "POST",
    body: form,
    query: { type: "local" },
    responseType: "json",
    signal,
    timeoutMs: 180_000,
  });
  const rawDocuments = asDocumentArray(envelope.data);
  if (envelope.code !== 0 && rawDocuments.length === 0) {
    unwrapPlatformEnvelope(envelope, { endpoint, httpStatus: 200 });
  }
  return {
    documents: rawDocuments.map((dto) => mapPlatformDocument(dto, datasetId)),
    partialFailure:
      envelope.code !== 0 && typeof envelope.message === "string"
        ? envelope.message
        : null,
  };
}

export function updateDatasetDocument(
  datasetId: string,
  documentId: string,
  payload: { name: string },
  signal?: AbortSignal,
): Promise<PlatformDatasetDocumentDto> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}`,
    { method: "PATCH", json: payload, signal },
  );
}

export function deleteDatasetDocuments(
  datasetId: string,
  documentIds: string[],
  signal?: AbortSignal,
): Promise<void> {
  return platformRequest(`/datasets/${encode(datasetId)}/documents`, {
    method: "DELETE",
    json: { ids: documentIds },
    signal,
  });
}

/** Active hybrid uses Python's ownership-checked canonical parse contract. */
export function parseDatasetDocuments(
  datasetId: string,
  documentIds: string[],
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/documents/parse`,
    {
      method: "POST",
      json: { document_ids: documentIds },
      signal,
    },
  );
}

/** The pinned Go image does not register this route, so hybrid safely falls back to Python. */
export function stopDatasetDocuments(
  datasetId: string,
  documentIds: string[],
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(
    `/datasets/${encode(datasetId)}/documents/stop`,
    {
      method: "POST",
      json: { document_ids: documentIds },
      signal,
    },
  );
}

export async function listDatasetIngestionTasks(
  datasetId: string,
  signal?: AbortSignal,
): Promise<PlatformIngestionTask[]> {
  const rows = await platformRequest<PlatformIngestionTaskDto[]>(
    "/datasets/ingestion/tasks",
    // Contract note: the active handler binds a JSON body on GET. Browsers do
    // not permit GET bodies, so callers must not use this until the runtime
    // accepts dataset_id as a query parameter. Retained as a typed contract for
    // non-browser tests and future-compatible runtimes.
    { query: { dataset_id: datasetId }, signal },
  );
  return (Array.isArray(rows) ? rows : [])
    .map(mapPlatformIngestionTask)
    .filter((row): row is PlatformIngestionTask => row !== null);
}

export function stopDatasetIngestionTasks(
  taskIds: string[],
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest("/datasets/ingestion/tasks", {
    method: "PUT",
    json: { tasks: taskIds },
    signal,
  });
}

export function removeDatasetIngestionTasks(
  taskIds: string[],
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest("/datasets/ingestion/tasks", {
    method: "DELETE",
    json: { tasks: taskIds },
    signal,
  });
}

export function createGenericDocument(
  payload: {
    name: string;
    kb_id: string;
    parser_id: string;
    created_by: string;
    type?: string;
    source?: string;
  },
  signal?: AbortSignal,
): Promise<PlatformGenericDocumentDto> {
  return platformRequest("/documents", { method: "POST", json: payload, signal });
}

/**
 * Typed form of the flat collection contract. Active v0.26.4 cannot satisfy
 * this in a browser because its handler performs ownership against an absent
 * dataset path parameter; the UI renders that runtime-disabled finding.
 */
export async function listGenericDocuments(
  signal?: AbortSignal,
): Promise<PlatformGenericDocumentDto[]> {
  const value = await platformRequest<unknown>("/documents", {
    responseType: "json",
    signal,
  });
  if (Array.isArray(value)) return value as PlatformGenericDocumentDto[];
  if (
    typeof value === "object" &&
    value !== null &&
    "data" in value &&
    Array.isArray(value.data)
  ) {
    return value.data as PlatformGenericDocumentDto[];
  }
  return [];
}

export function getGenericDocument(
  documentId: string,
  signal?: AbortSignal,
): Promise<PlatformGenericDocumentDto> {
  return platformRequest(`/documents/${encode(documentId)}`, {
    responseType: "json",
    signal,
  }).then((raw) => {
    if (
      typeof raw === "object" &&
      raw !== null &&
      "data" in raw &&
      typeof raw.data === "object" &&
      raw.data !== null
    ) {
      return raw.data as PlatformGenericDocumentDto;
    }
    return raw as PlatformGenericDocumentDto;
  });
}

export function updateGenericDocument(
  documentId: string,
  payload: {
    name?: string;
    run?: string;
    token_num?: number;
    chunk_num?: number;
    progress?: number;
    progress_msg?: string;
  },
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/documents/${encode(documentId)}`, {
    method: "PUT",
    json: payload,
    signal,
  });
}

export function deleteGenericDocument(
  documentId: string,
  signal?: AbortSignal,
): Promise<void> {
  return platformRequest(`/documents/${encode(documentId)}`, {
    method: "DELETE",
    signal,
  });
}

export function ingestGenericDocuments(
  documentIds: string[],
  run: string,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest("/documents/ingest", {
    method: "POST",
    json: { doc_ids: documentIds, run, delete: false, apply_kb: true },
    signal,
  });
}

export async function inspectDocumentUploads(
  files: File[],
  signal?: AbortSignal,
): Promise<PlatformUploadInspection[]> {
  const form = new FormData();
  for (const file of files) form.append("file", file, file.name);
  const value = await platformRequest<
    PlatformUploadInspection | PlatformUploadInspection[]
  >("/documents/upload", {
    method: "POST",
    body: form,
    signal,
    timeoutMs: 180_000,
  });
  return Array.isArray(value) ? value : [value];
}

async function fetchAsset(
  endpoint: string,
  signal?: AbortSignal,
): Promise<PlatformAsset> {
  let contentType = "application/octet-stream";
  let disposition: string | null = null;
  const blob = await platformRequest<Blob>(endpoint, {
    responseType: "blob",
    signal,
    getRetries: 0,
    onResponse: (response) => {
      contentType = response.headers.get("content-type") ?? contentType;
      disposition = response.headers.get("content-disposition");
    },
  });
  if (
    typeof blob !== "object" ||
    blob === null ||
    !("arrayBuffer" in blob) ||
    typeof blob.arrayBuffer !== "function"
  ) {
    throw new PlatformApiError("Belge içeriği ikili veri olarak alınamadı.", {
      httpStatus: 200,
      code: "INVALID_RESPONSE",
      endpoint,
    });
  }
  if (blob.size > PLATFORM_DOCUMENT_MAX_BYTES) {
    throw new PlatformApiError(
      "Belge içeriği güvenli görüntüleme sınırını aşıyor.",
      {
        httpStatus: 200,
        code: "RESPONSE_TOO_LARGE",
        endpoint,
      },
    );
  }
  return { blob, contentType, disposition };
}

export function fetchDocumentPreview(
  documentId: string,
  signal?: AbortSignal,
): Promise<PlatformAsset> {
  return fetchAsset(`/documents/${encode(documentId)}/preview`, signal);
}

export function downloadDatasetDocument(
  datasetId: string,
  documentId: string,
  signal?: AbortSignal,
): Promise<PlatformAsset> {
  return fetchAsset(
    `/datasets/${encode(datasetId)}/documents/${encode(documentId)}`,
    signal,
  );
}

export function fetchDocumentImage(
  imageId: string,
  signal?: AbortSignal,
): Promise<PlatformAsset> {
  return fetchAsset(`/documents/images/${encode(imageId)}`, signal);
}

export function fetchDocumentArtifact(
  filename: string,
  signal?: AbortSignal,
): Promise<PlatformAsset> {
  return fetchAsset(`/documents/artifact/${encode(filename)}`, signal);
}

export function listDocumentThumbnails(
  documentIds: string[],
  signal?: AbortSignal,
): Promise<Record<string, string | null>> {
  return platformRequest("/thumbnails", {
    query: { doc_ids: documentIds },
    signal,
  });
}

export function cancelPlatformTask(
  taskId: string,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest(`/tasks/${encode(taskId)}/cancel`, {
    method: "POST",
    signal,
  });
}

export function stopPlatformTask(
  taskId: string,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest(`/tasks/${encode(taskId)}`, {
    method: "PATCH",
    json: { action: "stop" },
    signal,
  });
}
