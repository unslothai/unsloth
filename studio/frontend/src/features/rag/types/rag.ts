// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface KnowledgeBase {
  id: string;
  name: string;
  description?: string | null;
  createdAt?: string | null;
  documentCount?: number;
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
  createdAt?: string | null;
}

/** Document upload response (KB or thread). */
export interface DocumentUploadResult {
  documentId: string;
  jobId: string;
  filename: string;
}

export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface IndexJob {
  id: string;
  documentId: string;
  status: JobStatus;
  stage?: string | null;
  progress?: number | null;
  error?: string | null;
}

/** One SSE frame from /jobs/{jobId}/events (before [DONE]). */
export interface JobEvent {
  type: "progress" | "complete" | "error";
  stage?: string | null;
  progress?: number | null;
  error?: string | null;
}

/** A chunk's highlight rect on a PDF page; coords 0..1, top-left origin. */
export interface PdfRegion {
  pageIndex: number;
  pageNumber: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Where a citation lives: page + highlight rects (PDF) or chunk text. */
export interface PreviewTarget {
  documentId: string;
  filename: string;
  mediaKind: "pdf" | "text";
  targetPage?: number | null;
  pdfRegions: PdfRegion[];
  text?: string | null;
}

/** File types the indexer accepts. */
export const RAG_UPLOAD_ACCEPT = ".pdf,.txt,.md,.markdown,.docx,.html,.htm";
