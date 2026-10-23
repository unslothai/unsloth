// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TEXT_ATTACHMENT_EXTENSIONS } from "../../chat/text-attachment-accept.ts";
import { RAG_DOCUMENT_UPLOAD_ACCEPT, RAG_UPLOAD_ACCEPT } from "../types/rag.ts";

// Everything the backend indexes (config.UPLOAD_EXTS), for every scope: chat, project and knowledge base.
export const RAG_SOURCE_UPLOAD_ACCEPT = [
  ...new Set([
    ...RAG_UPLOAD_ACCEPT.split(","),
    ...RAG_DOCUMENT_UPLOAD_ACCEPT.split(","),
    ...TEXT_ATTACHMENT_EXTENSIONS,
  ]),
].join(",");

export const SUPPORTED_SOURCES_HINT =
  "Supported: documents, spreadsheets, slides, e-books, email, text and code files.";

const ACCEPTED_UPLOAD_EXTS = new Set(RAG_SOURCE_UPLOAD_ACCEPT.split(","));

// `accept` only filters the picker, so a drop can carry anything, including an
// extension-less folder entry the backend would reject.
export function isSupportedSourceName(name: string): boolean {
  const dot = name.lastIndexOf(".");
  if (dot <= 0) return false;
  return ACCEPTED_UPLOAD_EXTS.has(name.slice(dot).toLowerCase());
}

/** Split a drop into what can be indexed and the names of what cannot, so the
 * caller can report the rejects instead of discarding them silently. */
export function partitionSupported<T>(
  entries: T[],
  nameOf: (entry: T) => string,
): { supported: T[]; unsupported: string[] } {
  const supported: T[] = [];
  const unsupported: string[] = [];
  for (const entry of entries) {
    const name = nameOf(entry);
    if (isSupportedSourceName(name)) supported.push(entry);
    else unsupported.push(name);
  }
  return { supported, unsupported };
}
