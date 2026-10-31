// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { documentKind, isMarkdown } from "@/components/file-viewer";
import { isTextAttachment } from "@/features/chat";
import { getDocumentFileUrl } from "../api/rag-api";
import type { RagDocument } from "../types/rag";

/** The stored source as a File, or null past `maxBytes` (0 = no cap). */
export async function fetchSourceFile(
  doc: Pick<RagDocument, "id" | "filename">,
  maxBytes: number,
): Promise<File | null> {
  const response = await fetch(await getDocumentFileUrl(doc.id));
  if (!response.ok) return null;
  if (maxBytes && Number(response.headers.get("content-length")) > maxBytes) {
    void response.body?.cancel();
    return null;
  }
  const blob = await response.blob();
  if (blob.size === 0 || (maxBytes && blob.size > maxBytes)) return null;
  // Linked-folder documents are named by their relative path.
  const name = doc.filename.split("/").pop() ?? doc.filename;
  // Most types are served as octet-stream; an empty type lets the extension decide the kind.
  const type = blob.type === "application/octet-stream" ? "" : blob.type;
  return new File([blob], name, { type });
}

/** Whether the attachment viewer can show this source: pages, markdown or text. */
export function isViewableSource(name: string): boolean {
  return (
    documentKind(name) !== null ||
    isMarkdown(name) ||
    isTextAttachment(name, undefined)
  );
}

/** Served as octet-stream: the viewer reads text only when the type says so. */
export function asViewerFile(file: File): File {
  const textual =
    documentKind(file.name, file.type) === null &&
    !isMarkdown(file.name, file.type);
  return textual ? new File([file], file.name, { type: "text/plain" }) : file;
}
