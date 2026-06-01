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
  RagSearchMode,
  RagSearchResult,
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

async function ragUpload(path: string, file: File): Promise<DocumentUploadResult> {
  const form = new FormData();
  form.append("file", file);
  // No Content-Type: let the browser set the multipart boundary.
  const response = await authFetch(`${RAG_BASE}${path}`, {
    method: "POST",
    body: form,
  });
  const json = await response.json().catch(() => null);
  if (!response.ok) throw new Error(parseErrorText(response.status, json));
  return json as DocumentUploadResult;
}

// ── Knowledge bases ──────────────────────────────────────────

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

// ── Documents ────────────────────────────────────────────────

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
): Promise<DocumentUploadResult> {
  return ragUpload(
    `/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
    file,
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
): Promise<DocumentUploadResult> {
  return ragUpload(`/threads/${encodeURIComponent(threadId)}/documents`, file);
}

export function deleteDocument(documentId: string): Promise<{ ok: boolean }> {
  return ragRequest(`/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });
}

// ── Index jobs ───────────────────────────────────────────────

export function getJob(jobId: string): Promise<IndexJob> {
  return ragRequest(`/jobs/${encodeURIComponent(jobId)}`);
}

/**
 * Stream indexing progress over SSE. Returns on `[DONE]`; transport errors
 * propagate so callers can fall back to polling getJob.
 */
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
}

// ── Preview (citation -> source location) ────────────────────

/** Resolve a citation to its source: page + highlight rects (PDF) or chunk text. */
export function getPreviewTarget(
  documentId: string,
  chunkId?: string,
): Promise<PreviewTarget> {
  const qs = chunkId ? `?chunk_id=${encodeURIComponent(chunkId)}` : "";
  return ragRequest(
    `/documents/${encodeURIComponent(documentId)}/preview-target${qs}`,
  );
}

/** Mint a short-lived signed URL (no bearer) so pdf.js can issue Range requests. */
export async function getDocumentFileUrl(documentId: string): Promise<string> {
  const data = await ragRequest<{ url: string }>(
    `/documents/${encodeURIComponent(documentId)}/file-url`,
  );
  return data.url;
}

// ── Search ───────────────────────────────────────────────────

export async function searchRag(payload: {
  query: string;
  kbId?: string;
  threadId?: string;
  topK?: number;
  minScore?: number;
  mode?: RagSearchMode;
}): Promise<RagSearchResult[]> {
  const data = await ragRequest<{ results: RagSearchResult[] }>("/search", {
    method: "POST",
    body: {
      query: payload.query,
      ...(payload.kbId ? { kb_id: payload.kbId } : {}),
      ...(payload.threadId ? { thread_id: payload.threadId } : {}),
      ...(payload.topK !== undefined ? { top_k: payload.topK } : {}),
      ...(payload.minScore !== undefined ? { min_score: payload.minScore } : {}),
      ...(payload.mode ? { mode: payload.mode } : {}),
    },
  });
  return data.results ?? [];
}
