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
  projectId?: string | null;
  createdAt?: string | null;
}

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

/** One SSE frame from /jobs/{jobId}/events. */
export interface JobEvent {
  type: "progress" | "complete" | "error";
  stage?: string | null;
  progress?: number | null;
  error?: string | null;
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

/** One figure rendered inline beneath the markdown preview. */
export interface PreviewFigure {
  id: string;
  page: number | null;
  caption: string | null;
  imageDataUrl?: string | null;
}

export interface PreviewTarget {
  documentId: string;
  filename: string;
  mediaKind: "pdf" | "text" | "markdown";
  targetPage?: number | null;
  pdfRegions: PdfRegion[];
  text?: string | null;
  /** Inline markdown body rendered for the "markdown" kind (extracted docs). */
  markdown?: string | null;
  /** Inline figures rendered beneath the markdown body. */
  figures?: PreviewFigure[];
}

export const RAG_UPLOAD_ACCEPT = ".pdf,.txt,.md,.markdown,.docx,.html,.htm";
