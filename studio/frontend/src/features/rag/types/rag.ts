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
  /** Size of the stored bytes; null when the file behind the row is gone. */
  sizeBytes?: number | null;
}

export function isLinkedFolderManaged(document: RagDocument): boolean {
  return Boolean(document.managed || document.linkedFolderId);
}

/** RagDocument enriched for the global uploaded-files list (settings Data tab). */
export interface UploadedDocument extends RagDocument {
  kbName?: string | null;
  projectName?: string | null;
}

export interface DocumentUploadResult {
  documentId: string;
  jobId: string;
  filename: string;
}

export type JobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type TerminalJobStatus = Extract<
  JobStatus,
  "completed" | "failed" | "cancelled"
>;

export function terminalJobStatus(status: JobStatus): TerminalJobStatus | null {
  return status === "completed" || status === "failed" || status === "cancelled"
    ? status
    : null;
}

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

export function retainActiveFolderJobs(
  folders: LinkedFolder[],
  jobs: Record<string, FolderSyncJob>,
): Record<string, FolderSyncJob> {
  const retained: Record<string, FolderSyncJob> = {};
  for (const folder of folders) {
    const job = jobs[folder.id];
    if (job && folder.activeJobId === job.id) retained[folder.id] = job;
  }
  return retained;
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

/** A source opened in the preview modal. The backend decides how to render it
 * and whether it may be edited, so the client keeps no list of file extensions
 * and the two can never disagree about what is editable. */
export interface DocumentContent {
  documentId: string;
  filename: string;
  /** "pdf" renders from the signed file URL and carries no text. */
  mediaKind: "pdf" | "text";
  /** What the View tab shows. "source" has no richer view, so the modal shows
   * the text alone with no toggle; the rest pair a View with an Edit. */
  preview: "source" | "markdown" | "html" | "extracted";
  text?: string | null;
  editable: boolean;
  /** Text was cut off at the size cap, or is not a faithful copy of the file, so
   * it is shown but not editable. */
  truncated: boolean;
  /** Why editing is unavailable; shown in the footer. Null when editable. */
  readOnlyReason?: string | null;
}

export const RAG_UPLOAD_ACCEPT = ".pdf,.txt,.md,.markdown,.docx,.html,.htm";
