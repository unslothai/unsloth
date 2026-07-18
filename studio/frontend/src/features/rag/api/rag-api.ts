// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";
import type {
  DocumentUploadResult,
  IndexJob,
  JobEvent,
  KnowledgeBase,
  PreviewTarget,
  RagDocument,
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

async function ragUpload(
  path: string,
  file: File,
  ocr?: boolean,
  caption?: boolean,
): Promise<DocumentUploadResult> {
  const form = new FormData();
  form.append("file", file);
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
  file: File,
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
  file: File,
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
  file: File,
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

export function deleteDocument(documentId: string): Promise<{ ok: boolean }> {
  return ragRequest(`/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });
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
          if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
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

export function getPreviewTarget(
  documentId: string,
  chunkId?: string,
): Promise<PreviewTarget> {
  const qs = chunkId ? `?chunk_id=${encodeURIComponent(chunkId)}` : "";
  return ragRequest(
    `/documents/${encodeURIComponent(documentId)}/preview-target${qs}`,
  );
}

// Signed URL (no bearer) so pdf.js can issue Range requests.
export async function getDocumentFileUrl(documentId: string): Promise<string> {
  const data = await ragRequest<{ url: string }>(
    `/documents/${encodeURIComponent(documentId)}/file-url`,
  );
  return data.url;
}
