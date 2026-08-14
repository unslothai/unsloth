


export interface KnowledgeBase {
  id: string;
  name: string;
  description?: string | null;
  createdAt?: string | null;
  updatedAt?: string | null;
  documentCount?: number;
  embeddingModel?: string;
  permission?: "me" | "team";
  chunkMethod?: string;
  parserConfig?: Record<string, unknown>;
  pipelineId?: string | null;
}

/** Index status: pending -> running -> completed | failed. */
export type DocumentStatus = "pending" | "running" | "completed" | "failed";

export interface RagDocument {
  id: string;
  filename: string;
  status: DocumentStatus;
  error?: string | null;
  numChunks?: number | null;
  kbId?: string | null;
  threadId?: string | null;
  projectId?: string | null;
  createdAt?: string | null;
}

/** RagDocument enriched for the global uploaded-files list (settings Data tab). */
export interface UploadedDocument extends RagDocument {
  sizeBytes?: number | null;
  kbName?: string | null;
  projectName?: string | null;
}

export interface DocumentUploadResult {
  documentId: string;
  jobId: string;
  filename: string;
}

/** Coords 0..1, top-left origin. */
export interface PdfRegion {
  pageIndex: number;
  pageNumber: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PreviewTarget {
  documentId: string;
  filename: string;
  mediaKind: "pdf" | "text";
  targetPage?: number | null;
  pdfRegions: PdfRegion[];
  text?: string | null;
}

export const RAG_UPLOAD_ACCEPT = ".pdf,.txt,.md,.markdown,.docx,.html,.htm";
