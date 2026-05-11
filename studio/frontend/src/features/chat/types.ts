// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CompleteAttachment,
  PendingAttachment,
} from "@assistant-ui/react";

export type ModelType = "base" | "lora" | "model1" | "model2";

export type ChatView =
  | { mode: "single"; threadId?: string; newThreadNonce?: string }
  | { mode: "compare"; pairId: string };

export interface ThreadRecord {
  id: string;
  title: string;
  modelType: ModelType;
  modelId?: string;
  pairId?: string;
  archived: boolean;
  createdAt: number;
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

/**
 * Client-side representation of a document the user has attached to the
 * composer but not yet sent.
 */
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
 * Discriminated attachment type for documents, extending assistant-ui's
 * PendingAttachment with document-specific fields. Replaces untyped
 * `as PendingAttachment` casts at the assistant-ui boundary.
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

/**
 * A DocumentPendingAttachment that has completed extraction and is ready
 * to be sent.
 */
export type DocumentCompleteAttachment = DocumentPendingAttachment & {
  status: { type: "complete" };
};

/**
 * Runtime type guard — narrows any assistant-ui attachment to
 * DocumentPendingAttachment. Use this instead of `as` casts.
 */
export function isDocumentAttachment(
  a: PendingAttachment | CompleteAttachment,
): a is DocumentPendingAttachment {
  return a.type === "document";
}

/**
 * Thrown when `send()` encounters a document attachment whose extracted
 * content has been lost (e.g. the File reference was not preserved). The
 * caller should mark the attachment incomplete and prompt the user to
 * re-attach.
 */
export class DocumentExtractionLostError extends Error {
  constructor() {
    super("Document extraction content is missing; re-attach the file.");
    this.name = "DocumentExtractionLostError";
  }
}
