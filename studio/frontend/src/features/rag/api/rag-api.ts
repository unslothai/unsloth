// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import type {
  DocumentUploadResult,
  FolderSyncJob,
  FolderSyncJobEvent,
  IndexJob,
  JobEvent,
  KnowledgeBase,
  LinkedFolder,
  LinkedFolderScope,
  PreviewTarget,
  RagDocument,
  UploadedDocument,
} from "../types/rag";

const RAG_BASE = "/api/rag";

function parseErrorText(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const { detail, message } = body as { detail?: unknown; message?: unknown };
    const formatted = formatFastApiDetail(detail);
    if (formatted) return formatted;
    if (typeof message === "string" && message) return message;
  }
  return `Request failed (${status})`;
}

async function ragRequest<T>(
  path: string,
  init?: { method?: string; body?: object },
): Promise<T> {
  const response = await authFetch(`${RAG_BASE}${path}`, {
    method: init?.method,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    body: init?.body ? JSON.stringify(init.body) : undefined,
  });
  if (response.status === 204) return undefined as T;
  const json = await response.json().catch(() => null);
  if (!response.ok) throw new Error(parseErrorText(response.status, json));
  return json as T;
}

/** A desktop drop the webview can only name through a Rust-signed grant. */
export interface NativeUploadRef {
  nativePathLease: string;
}

export type UploadSource = File | NativeUploadRef;

async function ragUpload(
  path: string,
  source: UploadSource,
  ocr?: boolean,
  caption?: boolean,
): Promise<DocumentUploadResult> {
  const form = new FormData();
  if (source instanceof File) form.append("file", source);
  else form.append("nativePathLease", source.nativePathLease);
  // Per-upload overrides for the vision passes; omitted -> backend config default.
  if (ocr !== undefined) form.append("ocr", String(ocr));
  if (caption !== undefined) form.append("caption", String(caption));
  // No Content-Type: let the browser set the multipart boundary.
  const response = await authFetch(`${RAG_BASE}${path}`, {
    method: "POST",
    body: form,
  });
  const json = await response.json().catch(() => null);
  if (!response.ok) throw new Error(parseErrorText(response.status, json));
  return json as DocumentUploadResult;
}

export async function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  const data = await ragRequest<{ knowledgeBases: KnowledgeBase[] }>(
    "/knowledge-bases",
  );
  return data.knowledgeBases ?? [];
}

export function createKnowledgeBase(payload: {
  name: string;
  description?: string;
}): Promise<{ id: string; name: string }> {
  return ragRequest("/knowledge-bases", {
    method: "POST",
    body: {
      name: payload.name,
      ...(payload.description ? { description: payload.description } : {}),
    },
  });
}

export function updateKnowledgeBase(
  kbId: string,
  payload: { name?: string; description?: string },
): Promise<{ ok: boolean }> {
  const body: Record<string, unknown> = {};
  if (payload.name !== undefined) body.name = payload.name;
  if (payload.description !== undefined) body.description = payload.description;
  return ragRequest(`/knowledge-bases/${encodeURIComponent(kbId)}`, {
    method: "PATCH",
    body,
  });
}

export function deleteKnowledgeBase(kbId: string): Promise<{ ok: boolean }> {
  return ragRequest(`/knowledge-bases/${encodeURIComponent(kbId)}`, {
    method: "DELETE",
  });
}

export async function listKnowledgeBaseDocuments(
  kbId: string,
): Promise<RagDocument[]> {
  const data = await ragRequest<{ documents: RagDocument[] }>(
    `/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
  );
  return data.documents ?? [];
}

export function uploadKnowledgeBaseDocument(
  kbId: string,
  file: UploadSource,
  ocr?: boolean,
  caption?: boolean,
): Promise<DocumentUploadResult> {
  return ragUpload(
    `/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
    file,
    ocr,
    caption,
  );
}

export async function listThreadDocuments(
  threadId: string,
): Promise<RagDocument[]> {
  const data = await ragRequest<{ documents: RagDocument[] }>(
    `/threads/${encodeURIComponent(threadId)}/documents`,
  );
  return data.documents ?? [];
}

export function uploadThreadDocument(
  threadId: string,
  file: UploadSource,
  ocr?: boolean,
  caption?: boolean,
): Promise<DocumentUploadResult> {
  return ragUpload(
    `/threads/${encodeURIComponent(threadId)}/documents`,
    file,
    ocr,
    caption,
  );
}

export async function listProjectDocuments(
  projectId: string,
): Promise<RagDocument[]> {
  const data = await ragRequest<{ documents: RagDocument[] }>(
    `/projects/${encodeURIComponent(projectId)}/documents`,
  );
  return data.documents ?? [];
}

export function uploadProjectDocument(
  projectId: string,
  file: UploadSource,
  ocr?: boolean,
  caption?: boolean,
): Promise<DocumentUploadResult> {
  return ragUpload(
    `/projects/${encodeURIComponent(projectId)}/documents`,
    file,
    ocr,
    caption,
  );
}

// Cached "does this project have indexed sources?" probe so the chat adapter can
// auto-scope project chats without a round trip per message. The sources panel
// invalidates on upload/delete.
const projectSourcesCache = new Map<string, { has: boolean; at: number }>();
const PROJECT_SOURCES_TTL_MS = 30_000;

export async function projectHasSources(projectId: string): Promise<boolean> {
  const cached = projectSourcesCache.get(projectId);
  if (cached && Date.now() - cached.at < PROJECT_SOURCES_TTL_MS) {
    return cached.has;
  }
  try {
    const docs = await listProjectDocuments(projectId);
    const has = docs.some((doc) => doc.status !== "failed");
    projectSourcesCache.set(projectId, { has, at: Date.now() });
    return has;
  } catch {
    // RAG unavailable or transient failure: don't cache, don't scope.
    return false;
  }
}

export function invalidateProjectSources(projectId: string): void {
  projectSourcesCache.delete(projectId);
}

export async function listLinkedFolders(
  scope?: LinkedFolderScope,
): Promise<LinkedFolder[]> {
  const query = scope
    ? `?scope_type=${encodeURIComponent(scope.type)}&scope_id=${encodeURIComponent(scope.id)}`
    : "";
  const data = await ragRequest<{ linkedFolders: LinkedFolder[] }>(
    `/linked-folders${query}`,
  );
  return data.linkedFolders ?? [];
}

interface LinkedFolderMutationResult {
  linkedFolder: LinkedFolder;
  job: FolderSyncJob;
}

export function createLinkedFolder(
  scope: LinkedFolderScope,
  nativePathLease: string,
  displayName: string,
): Promise<LinkedFolderMutationResult> {
  const parent =
    scope.type === "knowledge_base" ? "knowledge-bases" : "projects";
  return ragRequest(
    `/${parent}/${encodeURIComponent(scope.id)}/linked-folders`,
    {
      method: "POST",
      body: { nativePathLease, displayName },
    },
  );
}

export function deleteLinkedFolder(
  linkedFolderId: string,
  removeIndex: boolean,
): Promise<{ ok: boolean }> {
  return ragRequest(
    `/linked-folders/${encodeURIComponent(linkedFolderId)}?remove_index=${removeIndex}`,
    { method: "DELETE" },
  );
}

function startLinkedFolderJob(
  linkedFolderId: string,
  action: "sync" | "rebuild",
): Promise<{ job: FolderSyncJob }> {
  return ragRequest(
    `/linked-folders/${encodeURIComponent(linkedFolderId)}/${action}`,
    { method: "POST" },
  );
}

export function syncLinkedFolder(
  linkedFolderId: string,
): Promise<{ job: FolderSyncJob }> {
  return startLinkedFolderJob(linkedFolderId, "sync");
}

export function rebuildLinkedFolder(
  linkedFolderId: string,
): Promise<{ job: FolderSyncJob }> {
  return startLinkedFolderJob(linkedFolderId, "rebuild");
}

export function getFolderSyncJob(jobId: string): Promise<FolderSyncJob> {
  return ragRequest(`/linked-folder-jobs/${encodeURIComponent(jobId)}`);
}

export async function listAllDocuments(): Promise<UploadedDocument[]> {
  const data = await ragRequest<{ documents: UploadedDocument[] }>(
    "/documents",
  );
  return data.documents ?? [];
}

export async function deleteDocument(
  documentId: string,
  projectId?: string | null,
): Promise<{ ok: boolean }> {
  const result = await ragRequest<{ ok: boolean }>(
    `/documents/${encodeURIComponent(documentId)}`,
    {
      method: "DELETE",
    },
  );
  if (projectId) invalidateProjectSources(projectId);
  return result;
}

export function getJob(jobId: string): Promise<IndexJob> {
  return ragRequest(`/jobs/${encodeURIComponent(jobId)}`);
}

// SSE; returns on [DONE]. Transport errors propagate so callers can poll getJob.
export async function* streamJobEvents(
  jobId: string,
  signal?: AbortSignal,
): AsyncGenerator<JobEvent> {
  const response = await authFetch(
    `${RAG_BASE}/jobs/${encodeURIComponent(jobId)}/events`,
    signal ? { signal } : undefined,
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(parseErrorText(response.status, body));
  }
  if (!response.body) throw new Error("Stream response missing body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let separatorIndex = buffer.search(/\r?\n\r?\n/);
      while (separatorIndex >= 0) {
        const rawEvent = buffer.slice(0, separatorIndex);
        const separatorLength = buffer[separatorIndex] === "\r" ? 4 : 2;
        buffer = buffer.slice(separatorIndex + separatorLength);

        const dataLines: string[] = [];
        for (const line of rawEvent.split(/\r?\n/)) {
          if (line.startsWith("data:"))
            dataLines.push(line.slice(5).trimStart());
        }
        if (dataLines.length > 0) {
          const dataText = dataLines.join("\n");
          if (dataText === "[DONE]") return;
          try {
            yield JSON.parse(dataText) as JobEvent;
          } catch {
            // Ignore unparseable frames; [DONE] still ends the loop.
          }
        }
        separatorIndex = buffer.search(/\r?\n\r?\n/);
      }
    }
  } finally {
    // Release the stream lock now instead of leaking the reader until GC.
    try {
      await reader.cancel();
    } catch {
      // already closed
    }
  }
}

export async function* streamFolderSyncJobEvents(
  jobId: string,
  signal?: AbortSignal,
): AsyncGenerator<FolderSyncJobEvent> {
  const response = await authFetch(
    `${RAG_BASE}/linked-folder-jobs/${encodeURIComponent(jobId)}/events`,
    signal ? { signal } : undefined,
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(parseErrorText(response.status, body));
  }
  if (!response.body) throw new Error("Stream response missing body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let separatorIndex = buffer.search(/\r?\n\r?\n/);
      while (separatorIndex >= 0) {
        const rawEvent = buffer.slice(0, separatorIndex);
        const separatorLength = buffer[separatorIndex] === "\r" ? 4 : 2;
        buffer = buffer.slice(separatorIndex + separatorLength);
        const data = rawEvent
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (data === "[DONE]") return;
        if (data) {
          try {
            yield JSON.parse(data) as FolderSyncJobEvent;
          } catch {
            // Ignore malformed frames and continue to the terminal event.
          }
        }
        separatorIndex = buffer.search(/\r?\n\r?\n/);
      }
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      // already closed
    }
  }
}

export function getPreviewTarget(
  documentId: string,
  chunkId?: string,
): Promise<PreviewTarget> {
  const qs = chunkId ? `?chunk_id=${encodeURIComponent(chunkId)}` : "";
  return ragRequest(
    `/documents/${encodeURIComponent(documentId)}/preview-target${qs}`,
  );
}

// Signed URL (no bearer) so pdf.js can issue Range requests. Absolute because
// consumers bypass authFetch, and a relative path under Tauri resolves against
// the webview origin.
export async function getDocumentFileUrl(documentId: string): Promise<string> {
  const data = await ragRequest<{ url: string }>(
    `/documents/${encodeURIComponent(documentId)}/file-url`,
  );
  return apiUrl(data.url);
}
