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
  linkedFolderId?: string | null;
  managed: boolean;
  createdAt?: string | null;
}

export function isLinkedFolderManaged(document: RagDocument): boolean {
  return Boolean(document.managed || document.linkedFolderId);
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

export type JobStatus = "pending" | "running" | "completed" | "failed";

export interface IndexJob {
  id: string;
  documentId: string;
  status: JobStatus;
  stage?: string | null;
  progress?: number | null;
  error?: string | null;
  numChunks?: number | null;
}

/** One SSE frame from /jobs/{jobId}/events. */
export interface JobEvent {
  type: "progress" | "complete" | "error";
  stage?: string | null;
  progress?: number | null;
  error?: string | null;
  num_chunks?: number | null;
}

export type LinkedFolderScopeType = "knowledge_base" | "project";

export interface LinkedFolderScope {
  type: LinkedFolderScopeType;
  id: string;
}

export type LinkedFolderStatus = "idle" | "syncing" | "error";

/** A local directory whose durable access grant is held by the desktop backend. */
export interface LinkedFolder {
  id: string;
  displayName: string;
  scopeType: LinkedFolderScopeType;
  scopeId: string;
  scopeName?: string | null;
  status: LinkedFolderStatus;
  documentCount?: number;
  lastSyncedAt?: string | null;
  error?: string | null;
  activeJobId?: string | null;
  createdAt?: string | null;
}

export function linkedFolderSourcesChanged(
  previous: LinkedFolder[] | null,
  current: LinkedFolder[],
): boolean {
  if (!previous) return false;
  const previousById = new Map(previous.map((folder) => [folder.id, folder]));
  if (
    previous.length !== current.length ||
    previous.some((folder) => !current.some((row) => row.id === folder.id))
  ) {
    return true;
  }
  return current.some((folder) => {
    const prior = previousById.get(folder.id);
    return (
      prior !== undefined &&
      (prior.documentCount !== folder.documentCount ||
        prior.lastSyncedAt !== folder.lastSyncedAt)
    );
  });
}

export type FolderSyncMode = "sync" | "rebuild";

/** Aggregate job for discovering and indexing all changes in a linked folder. */
export interface FolderSyncJob {
  id: string;
  linkedFolderId: string;
  mode: FolderSyncMode;
  status: JobStatus;
  stage?: string | null;
  progress?: number | null;
  discoveredFiles?: number;
  processedFiles?: number;
  indexedFiles?: number;
  removedFiles?: number;
  failedFiles?: number;
  error?: string | null;
}

/** One SSE frame from /linked-folder-jobs/{jobId}/events. */
export interface FolderSyncJobEvent extends Partial<FolderSyncJob> {
  type: "progress" | "complete" | "error";
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
