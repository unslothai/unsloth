// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";

export type ChunkingStrategy = "standard" | "late";
export type KBMode = "text" | "multimodal";

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string | null;
  embedding_model: string;
  chunking_strategy: ChunkingStrategy;
  mode: KBMode;
  created_at: number;
}

export interface RagDocument {
  id: string;
  kb_id: string | null;
  thread_id: string | null;
  filename: string;
  content_type: string | null;
  status: "pending" | "running" | "completed" | "failed";
  num_chunks: number;
  byte_size: number;
  error: string | null;
  created_at: number;
}

export interface UploadResponse {
  document_id: string;
  job_id: string;
  filename: string;
}

export interface SearchHit {
  chunk_id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  score: number;
  page_number: number | null;
  filename: string | null;
  kind?: "text" | "image" | "caption";
  image_url?: string | null;
}

export interface SearchRequest {
  query: string;
  kb_id?: string;
  thread_id?: string;
  top_k?: number;
  mode?: "bm25" | "dense" | "hybrid";
  document_ids?: string[];
  enable_rerank?: boolean;
  reranker_model?: string;
  /** Cosine-similarity floor (0..1). Hits below are dropped server-side. */
  min_score?: number;
}

export type JobEvent =
  | { type: "status"; status: string; stage?: string | null; progress?: number; error?: string | null }
  | { type: "progress"; stage: string; progress: number }
  | { type: "complete"; num_chunks: number }
  | { type: "error"; error: string };

function parseErrorText(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const detail = (body as { detail?: unknown }).detail;
    const formatted = formatFastApiDetail(detail);
    if (formatted) return formatted;
  }
  return `Request failed (${status})`;
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(parseErrorText(response.status, body));
  }
  return body as T;
}

async function throwOnError(response: Response): Promise<void> {
  if (response.ok) return;
  const body = await response.json().catch(() => null);
  throw new Error(parseErrorText(response.status, body));
}

// --- Knowledge bases ---

export async function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  const response = await authFetch("/api/rag/knowledge-bases");
  const body = await parseJsonOrThrow<{ knowledge_bases: KnowledgeBase[] }>(response);
  return body.knowledge_bases;
}

export interface CreateKnowledgeBaseRequest {
  name: string;
  description?: string;
  embedding_model?: string;
  chunking_strategy?: ChunkingStrategy;
  mode?: KBMode;
}

export async function createKnowledgeBase(
  req: CreateKnowledgeBaseRequest,
): Promise<KnowledgeBase> {
  const response = await authFetch("/api/rag/knowledge-bases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return parseJsonOrThrow<KnowledgeBase>(response);
}

export async function deleteKnowledgeBase(kbId: string): Promise<void> {
  const response = await authFetch(
    `/api/rag/knowledge-bases/${encodeURIComponent(kbId)}`,
    { method: "DELETE" },
  );
  await throwOnError(response);
}

// --- Documents ---

export async function listKBDocuments(kbId: string): Promise<RagDocument[]> {
  const response = await authFetch(
    `/api/rag/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
  );
  const body = await parseJsonOrThrow<{ documents: RagDocument[] }>(response);
  return body.documents;
}

export async function listThreadDocuments(threadId: string): Promise<RagDocument[]> {
  const response = await authFetch(
    `/api/rag/threads/${encodeURIComponent(threadId)}/documents`,
  );
  const body = await parseJsonOrThrow<{ documents: RagDocument[] }>(response);
  return body.documents;
}

export async function uploadKBDocument(
  kbId: string,
  file: File,
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await authFetch(
    `/api/rag/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
    { method: "POST", body: form },
  );
  return parseJsonOrThrow<UploadResponse>(response);
}

export async function uploadThreadDocument(
  threadId: string,
  file: File,
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await authFetch(
    `/api/rag/threads/${encodeURIComponent(threadId)}/documents`,
    { method: "POST", body: form },
  );
  return parseJsonOrThrow<UploadResponse>(response);
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await authFetch(
    `/api/rag/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  );
  await throwOnError(response);
}

export interface ThreadIndexSummary {
  thread_id: string;
  title: string | null;
  num_documents: number;
  num_chunks: number;
}

export async function listThreadIndexes(): Promise<ThreadIndexSummary[]> {
  const response = await authFetch("/api/rag/thread-indexes");
  const body = await parseJsonOrThrow<{ threads: ThreadIndexSummary[] }>(response);
  return body.threads;
}

export async function clearThreadDocuments(threadId: string): Promise<void> {
  const response = await authFetch(
    `/api/rag/threads/${encodeURIComponent(threadId)}/documents`,
    { method: "DELETE" },
  );
  await throwOnError(response);
}

export interface ReingestResponse {
  job_ids: string[];
  document_ids: string[];
}

export interface ReingestKBOptions {
  chunking_strategy?: ChunkingStrategy;
  mode?: KBMode;
  embedding_model?: string;
}

export async function reingestKnowledgeBase(
  kbId: string,
  opts: ReingestKBOptions = {},
): Promise<ReingestResponse> {
  const response = await authFetch(
    `/api/rag/knowledge-bases/${encodeURIComponent(kbId)}/reingest`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts),
    },
  );
  return parseJsonOrThrow<ReingestResponse>(response);
}

export interface ThreadRagSettings {
  chunking_strategy: ChunkingStrategy;
  mode: KBMode;
  embedding_model: string | null;
}

export interface UpdateThreadRagSettingsRequest {
  chunking_strategy?: ChunkingStrategy;
  mode?: KBMode;
  embedding_model?: string | null;
}

export async function getThreadRagSettings(
  threadId: string,
): Promise<ThreadRagSettings> {
  const response = await authFetch(
    `/api/rag/threads/${encodeURIComponent(threadId)}/settings`,
  );
  return parseJsonOrThrow<ThreadRagSettings>(response);
}

export async function setThreadRagSettings(
  threadId: string,
  payload: UpdateThreadRagSettingsRequest,
): Promise<ThreadRagSettings> {
  const response = await authFetch(
    `/api/rag/threads/${encodeURIComponent(threadId)}/settings`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
  return parseJsonOrThrow<ThreadRagSettings>(response);
}

export async function reingestThreadDocuments(
  threadId: string,
  opts: UpdateThreadRagSettingsRequest = {},
): Promise<ReingestResponse> {
  const response = await authFetch(
    `/api/rag/threads/${encodeURIComponent(threadId)}/reingest`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts),
    },
  );
  return parseJsonOrThrow<ReingestResponse>(response);
}

export interface RagDefaults {
  chunking_strategy: ChunkingStrategy;
  mode: KBMode;
  embedding_model: string | null;
}

export async function getRagDefaults(): Promise<RagDefaults> {
  const response = await authFetch("/api/rag/defaults");
  return parseJsonOrThrow<RagDefaults>(response);
}

export interface UpdateRagDefaultsRequest {
  chunking_strategy?: ChunkingStrategy;
  mode?: KBMode;
  embedding_model?: string | null;
}

export async function setRagDefaults(
  payload: UpdateRagDefaultsRequest,
): Promise<RagDefaults> {
  const response = await authFetch("/api/rag/defaults", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJsonOrThrow<RagDefaults>(response);
}

/** Preload the configured embedder on the backend. Long-running (cold
 *  load can take 30s+). Fire-and-forget: failure is non-fatal because
 *  the first real query will lazy-load again. */
export async function warmupRagEmbedder(): Promise<void> {
  await authFetch("/api/rag/warmup", { method: "POST" });
}

// --- Search ---

export async function search(req: SearchRequest): Promise<SearchHit[]> {
  const response = await authFetch("/api/rag/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  const body = await parseJsonOrThrow<{ hits: SearchHit[] }>(response);
  return body.hits;
}

// --- Ingestion SSE ---

/** Subscribe to a job's SSE stream; returns an unsubscribe fn.
 *  Token goes via ?token= since EventSource cannot send headers. */
export function subscribeToJobEvents(
  jobId: string,
  handlers: {
    onEvent?: (event: JobEvent) => void;
    onError?: (error: Error) => void;
    onClose?: () => void;
  },
): () => void {
  const token = getAuthToken();
  const params = token ? `?token=${encodeURIComponent(token)}` : "";
  const url = apiUrl(`/api/rag/jobs/${encodeURIComponent(jobId)}/events${params}`);
  const source = new EventSource(url);

  source.onmessage = (e) => {
    try {
      const parsed = JSON.parse(e.data) as JobEvent;
      handlers.onEvent?.(parsed);
      if (parsed.type === "complete" || parsed.type === "error") {
        source.close();
        handlers.onClose?.();
      }
    } catch (err) {
      handlers.onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  };

  source.onerror = () => {
    // Browser auto-reconnects unless we close; let the consumer decide instead.
    source.close();
    handlers.onError?.(new Error("SSE connection lost"));
    handlers.onClose?.();
  };

  return () => {
    source.close();
    handlers.onClose?.();
  };
}
