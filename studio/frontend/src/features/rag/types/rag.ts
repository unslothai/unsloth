// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Retrieval source the search_knowledge_base tool runs against. */
export type RagSourceType = "kb" | "thread";

/** Search backends the retriever can run a query through. */
export type RagSearchMode = "hybrid" | "lexical" | "dense";

export interface KnowledgeBase {
  id: string;
  name: string;
  description?: string | null;
  createdAt?: string | null;
  documentCount?: number;
}

/** Document index status. pending -> running -> completed | failed. */
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

/** Response from a document upload (KB or per-thread). */
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

/** A single SSE frame from /jobs/{jobId}/events (before the [DONE] sentinel). */
export interface JobEvent {
  type: "progress" | "complete" | "error";
  stage?: string | null;
  progress?: number | null;
  error?: string | null;
}

export interface RagSearchResult {
  chunkId: string;
  documentId: string;
  filename: string;
  page?: number | null;
  score: number;
  text: string;
}

/**
 * rag_scope forwarded on the chat request when retrieval is enabled. Exactly
 * one of kb_id / thread_id is set: kb_id when a knowledge base is selected,
 * thread_id when the source is the current thread's own documents.
 */
export interface RagScope {
  kb_id?: string;
  thread_id?: string;
  default_top_k: number;
  min_score: number;
  mode: RagSearchMode;
}

/** Upload types the indexer accepts. */
export const RAG_UPLOAD_ACCEPT = ".pdf,.txt,.md,.markdown,.docx,.html,.htm";
