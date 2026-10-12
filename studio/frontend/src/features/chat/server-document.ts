// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { extractDocumentText } from "@/features/rag/api/rag-api";
import { MAX_SERVER_DOCUMENT_BYTES } from "./open-document-accept";
import { MAX_TEXT_ATTACHMENT_BYTES } from "./text-attachment-accept";

/** Parsed fine, but nothing to inline: an image-only .epub or .odp. */
export class NoDocumentTextError extends Error {}

// Per File, so the chip's preview reuses the upload the adapter already made.
const extracted = new WeakMap<File, Promise<string>>();

async function extract(file: File): Promise<string> {
  if (file.size > MAX_SERVER_DOCUMENT_BYTES) {
    throw new Error(`File is too large: ${file.name}`);
  }
  let result: { text: string; truncated: boolean };
  try {
    result = await extractDocumentText(file);
  } catch (cause) {
    throw new Error(
      (cause as Error | undefined)?.message ||
        `File could not be read: ${file.name}`,
    );
  }
  if (!result.text.trim()) {
    throw new NoDocumentTextError(`${file.name} has no text to read.`);
  }
  return result.truncated
    ? `${result.text}\n\n[Truncated: the text stops after ${MAX_TEXT_ATTACHMENT_BYTES.toLocaleString("en-US")} bytes.]`
    : result.text;
}

/** The backend parser's text for a document the browser readers cannot open. */
export function extractServerDocumentText(file: File): Promise<string> {
  let pending = extracted.get(file);
  if (!pending) {
    pending = extract(file);
    extracted.set(file, pending);
    // A failed read (offline, signed out) may succeed when attached again.
    pending.catch(() => extracted.delete(file));
  }
  return pending;
}
