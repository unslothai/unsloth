export type PlatformDocumentStatus =
  | "pending"
  | "running"
  | "completed"
  | "cancelled"
  | "failed";

export interface PlatformDatasetDocumentDto {
  id?: unknown;
  dataset_id?: unknown;
  name?: unknown;
  thumbnail?: unknown;
  size?: unknown;
  type?: unknown;
  created_by?: unknown;
  location?: unknown;
  token_count?: unknown;
  chunk_count?: unknown;
  progress?: unknown;
  progress_msg?: unknown;
  process_begin_at?: unknown;
  process_duration?: unknown;
  suffix?: unknown;
  run?: unknown;
  status?: unknown;
  parser_id?: unknown;
  chunk_method?: unknown;
  pipeline_id?: unknown;
  pipeline_name?: unknown;
  nickname?: unknown;
  parser_config?: unknown;
  meta_fields?: unknown;
  create_time?: unknown;
  create_date?: unknown;
  update_time?: unknown;
  update_date?: unknown;
}

export interface PlatformDocument {
  id: string;
  datasetId: string;
  name: string;
  thumbnail: string | null;
  sizeBytes: number;
  sourceType: string;
  location: string | null;
  tokenCount: number;
  chunkCount: number;
  progress: number;
  progressMessage: string | null;
  processDuration: number;
  suffix: string;
  run: string;
  backendStatus: string;
  status: PlatformDocumentStatus;
  parserId: string;
  chunkMethod: string;
  pipelineId: string | null;
  pipelineName: string | null;
  createdAt: string | null;
  updatedAt: string | null;
}

export interface PlatformDatasetDocumentListResult {
  items: PlatformDocument[];
  total: number;
}

export interface PlatformDocumentUploadResult {
  documents: PlatformDocument[];
  partialFailure: string | null;
}

export interface PlatformIngestionTaskDto {
  id?: unknown;
  document_id?: unknown;
  dataset_id?: unknown;
  status?: unknown;
  create_time?: unknown;
  update_time?: unknown;
}

export interface PlatformIngestionTask {
  id: string;
  documentId: string;
  datasetId: string;
  status: string;
}

export interface PlatformUploadInspection {
  name?: unknown;
  type?: unknown;
  size?: unknown;
  suffix?: unknown;
  [key: string]: unknown;
}

export interface PlatformGenericDocumentDto {
  id?: unknown;
  name?: unknown;
  kb_id?: unknown;
  parser_id?: unknown;
  pipeline_id?: unknown;
  type?: unknown;
  source_type?: unknown;
  created_by?: unknown;
  location?: unknown;
  size?: unknown;
  token_num?: unknown;
  chunk_num?: unknown;
  progress?: unknown;
  progress_msg?: unknown;
  process_duration?: unknown;
  suffix?: unknown;
  run?: unknown;
  status?: unknown;
  created_at?: unknown;
  updated_at?: unknown;
}

export interface PlatformAsset {
  blob: Blob;
  contentType: string;
  disposition: string | null;
}

export interface PlatformFileValidationFailure {
  file: File;
  reason: "empty" | "too-large" | "unsupported" | "unsafe-name";
  message: string;
}

export const PLATFORM_DOCUMENT_MAX_BYTES = 128 * 1024 * 1024;

const SUPPORTED_DOCUMENT_SUFFIXES = new Set([
  "bmp",
  "csv",
  "doc",
  "docx",
  "gif",
  "htm",
  "html",
  "jpeg",
  "jpg",
  "json",
  "jsonl",
  "md",
  "pdf",
  "png",
  "ppt",
  "pptx",
  "text",
  "tif",
  "tiff",
  "txt",
  "webp",
  "xls",
  "xlsx",
]);

function stringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function finiteNumber(value: unknown, fallback = 0): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function optionalString(value: unknown): string | null {
  const result = stringValue(value);
  return result || null;
}

function dateValue(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) return value;
  const timestamp = finiteNumber(value, 0);
  if (timestamp <= 0) return null;
  const millis = timestamp > 10_000_000_000 ? timestamp : timestamp * 1000;
  return new Date(millis).toISOString();
}

export function platformDocumentStatus(
  runValue: unknown,
  statusValue?: unknown,
): PlatformDocumentStatus {
  const value = stringValue(runValue || statusValue).toUpperCase();
  if (["1", "RUNNING", "SCHEDULE", "PENDING", "QUEUED"].includes(value)) {
    return value === "PENDING" || value === "QUEUED"
      ? "pending"
      : "running";
  }
  if (["2", "CANCEL", "CANCELLED", "CANCELED"].includes(value)) {
    return "cancelled";
  }
  if (["3", "DONE", "COMPLETED", "SUCCESS"].includes(value)) {
    return "completed";
  }
  if (["4", "FAIL", "FAILED", "ERROR"].includes(value)) return "failed";
  return "pending";
}

export function mapPlatformDocument(
  dto: PlatformDatasetDocumentDto,
  fallbackDatasetId = "",
): PlatformDocument {
  const id = stringValue(dto.id);
  const name = stringValue(dto.name);
  if (!id || !name) {
    throw new TypeError("Rag Platform belge yanıtında id veya ad eksik.");
  }
  const progress = Math.min(1, Math.max(0, finiteNumber(dto.progress)));
  return {
    id,
    datasetId: stringValue(dto.dataset_id) || fallbackDatasetId,
    name,
    thumbnail: optionalString(dto.thumbnail),
    sizeBytes: Math.max(0, finiteNumber(dto.size)),
    sourceType: stringValue(dto.type) || "unknown",
    location: optionalString(dto.location),
    tokenCount: Math.max(0, finiteNumber(dto.token_count)),
    chunkCount: Math.max(0, finiteNumber(dto.chunk_count)),
    progress,
    progressMessage: optionalString(dto.progress_msg),
    processDuration: Math.max(0, finiteNumber(dto.process_duration)),
    suffix: stringValue(dto.suffix).replace(/^\./, "").toLowerCase(),
    run: stringValue(dto.run),
    backendStatus: stringValue(dto.status),
    status: platformDocumentStatus(dto.run, dto.status),
    parserId: stringValue(dto.parser_id),
    chunkMethod: stringValue(dto.chunk_method),
    pipelineId: optionalString(dto.pipeline_id),
    pipelineName: optionalString(dto.pipeline_name),
    createdAt: dateValue(dto.create_date ?? dto.create_time),
    updatedAt: dateValue(dto.update_date ?? dto.update_time),
  };
}

export function mapPlatformIngestionTask(
  dto: PlatformIngestionTaskDto,
): PlatformIngestionTask | null {
  const id = stringValue(dto.id);
  const documentId = stringValue(dto.document_id);
  if (!id || !documentId) return null;
  return {
    id,
    documentId,
    datasetId: stringValue(dto.dataset_id),
    status: stringValue(dto.status),
  };
}

export function validatePlatformDocumentFile(
  file: File,
): PlatformFileValidationFailure | null {
  if (!file.name || hasControlCharacters(file.name)) {
    return { file, reason: "unsafe-name", message: "Dosya adı güvenli değil." };
  }
  if (new TextEncoder().encode(file.name).length > 255) {
    return {
      file,
      reason: "unsafe-name",
      message: "Dosya adı 255 bayttan uzun olamaz.",
    };
  }
  if (file.size <= 0) {
    return { file, reason: "empty", message: "Boş dosyalar yüklenemez." };
  }
  if (file.size > PLATFORM_DOCUMENT_MAX_BYTES) {
    return {
      file,
      reason: "too-large",
      message: "Dosya 128 MB sınırını aşıyor.",
    };
  }
  const suffix = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!SUPPORTED_DOCUMENT_SUFFIXES.has(suffix)) {
    return {
      file,
      reason: "unsupported",
      message: `.${suffix || "?"} biçimi desteklenmiyor.`,
    };
  }
  return null;
}

export function hasControlCharacters(value: string): boolean {
  return Array.from(value).some((character) => {
    const codePoint = character.codePointAt(0) ?? 0;
    return codePoint <= 31 || codePoint === 127;
  });
}

export function isInlineSafeContentType(contentType: string): boolean {
  const normalized = contentType.split(";", 1)[0]?.trim().toLowerCase();
  return (
    normalized === "application/pdf" ||
    normalized === "text/plain" ||
    normalized === "text/markdown" ||
    normalized?.startsWith("image/") === true
  );
}
