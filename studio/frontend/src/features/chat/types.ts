// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CompleteAttachment,
  PendingAttachment,
} from "@assistant-ui/react";

export type ModelType = "base" | "lora" | "model1" | "model2";

export type ChatView =
  | {
      mode: "project";
      projectId: string;
    }
  | {
      mode: "single";
      threadId?: string;
      newThreadNonce?: string;
      projectId?: string | null;
    }
  | { mode: "compare"; pairId: string; projectId?: string | null };

export interface ProjectRecord {
  id: string;
  name: string;
  instructions?: string;
  rootPath?: string | null;
  sandboxPath?: string | null;
  archived: boolean;
  createdAt: number;
  updatedAt: number;
}

export interface ThreadRecord {
  id: string;
  title: string;
  modelType: ModelType;
  modelId?: string;
  pairId?: string;
  projectId?: string | null;
  archived: boolean;
  createdAt: number;
  /**
   * OpenAI shell tool container id from a prior response. When set, the
   * next turn reuses it via `environment.type="container_reference"` so
   * the model can read files it wrote earlier; else auto-creates one.
   *
   * Containers expire after ~20 min idle; on a stale id the backend
   * emits `_toolEvent.type="container_invalidated"` and the chat-adapter
   * clears this field so the next turn falls back to auto-create.
   */
  openaiCodeExecContainerId?: string | null;
  /**
   * Anthropic code_execution container id from a prior response. When
   * set, the next turn sends a top-level `container` on /v1/messages so
   * filesystem state (files, packages, variables) persists; else auto-
   * creates one.
   *
   * Containers expire after ~1 hour; on a stale id the backend emits
   * `_toolEvent.type="container_invalidated"` and the chat-adapter
   * clears this field so the next turn falls back to auto-create.
   */
  anthropicCodeExecContainerId?: string | null;
  /**
   * If this thread was created via fork-from-message, points back at
   * the source thread + branch-point msg. Null/undefined for non-fork
   * threads. Used by the sidebar "fork" badge and the parent thread's
   * "N forks" indicator on the branch-point msg.
   */
  forkedFromThreadId?: string | null;
  forkedFromMessageId?: string | null;
}

export interface MessageRecord {
  id: string;
  threadId: string;
  parentId?: string | null;
  role: import("@assistant-ui/react").ThreadMessage["role"];
  content: import("@assistant-ui/react").ThreadMessage["content"];
  attachments?: import("@assistant-ui/react").ThreadMessage["attachments"];
  metadata?: Record<string, unknown>;
  createdAt: number;
}

/** One figure discovered in an uploaded document. */
export interface ExtractedFigure {
  id: string;
  page: number | null;
  caption: string | null;
  error: string | null;
  kind?: "figure" | "page";
  image_mime?: string | null;
  image_base64?: string | null;
  image_width?: number | null;
  image_height?: number | null;
}

/** Shape returned by POST /api/inference/chat/extract-document. */
export interface ExtractedDocument {
  schema_version?: 1;
  filename: string;
  markdown: string;
  page_count: number;
  tokens_est: number;
  truncated?: boolean;
  figures: ExtractedFigure[];
  describe_skipped_reason: string | null;
  /** Backend that served describe calls: 'gguf' | 'transformers' | 'unsloth' | 'none'. */
  vlm_source?: string | null;
  /** Identifier of the VLM whose captions appear in this document. */
  vlm_model?: string | null;
  /** Whether the active model can receive an extracted visual payload. */
  image_input_available: boolean;
  warnings: string[];
}

/** Runtime probe for the currently-loaded vision model. */
export interface VlmCapabilityInfo {
  is_vlm: boolean;
  endpoint_url: string | null;
  model_name: string | null;
  source: "gguf" | "transformers" | "unsloth" | "none";
  reason: string | null;
}

/** Shape returned by GET /api/inference/chat/document-support. */
export interface DocumentSupport {
  schema_version?: 1;
  extraction_available: boolean;
  max_visual_payloads: number;
  max_extract_concurrency?: number;
  format_support?: Record<string, boolean>;
  unavailable_formats?: Record<string, string>;
  vlm: VlmCapabilityInfo;
}

export type DocumentExtractionErrorCode =
  | "oversized"
  | "unsupported_type"
  | "network"
  | "unauthorized"
  | "extractor_unavailable"
  | "encrypted"
  | "timeout"
  | "busy"
  | "client_closed"
  | "extraction_failed"
  | "aborted";

/** A document attached to the composer but not yet sent. */
export interface PendingDocumentAttachment {
  id: string;
  filename: string;
  sizeBytes: number;
  document: ExtractedDocument;
  extractedAt: number;
  truncated?: boolean;
  sentImageIndexes?: number[];
}

/**
 * Document attachment extending assistant-ui's PendingAttachment with
 * document fields. Replaces untyped `as PendingAttachment` casts at the assistant-ui boundary.
 */
export interface DocumentPendingAttachment extends PendingAttachment {
  type: "document";
  file: File;
  document?: ExtractedDocument;
  sizeBytes: number;
  extractedAt: number;
  truncated?: boolean;
  sentImageIndexes?: number[];
  errorCode?: DocumentExtractionErrorCode;
  errorMessage?: string;
  retryCount?: number;
}

/** Narrows an assistant-ui attachment to DocumentPendingAttachment. */
export function isDocumentAttachment(
  a: PendingAttachment | CompleteAttachment,
): a is DocumentPendingAttachment {
  return a.type === "document";
}

/** Thrown when `send()` finds a document attachment whose extracted content
 * was lost; the caller marks it incomplete and prompts a re-attach. */
export class DocumentExtractionLostError extends Error {
  constructor() {
    super("Document extraction content is missing; re-attach the file.");
    this.name = "DocumentExtractionLostError";
  }
}
