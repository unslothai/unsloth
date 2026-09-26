// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The document types rendered as pages or a grid rather than as text. */
export type DocumentKind = "pdf" | "docx" | "sheet" | "slides";

const EXTENSION_KINDS: Record<string, DocumentKind> = {
  pdf: "pdf",
  docx: "docx",
  xlsx: "sheet",
  xlsm: "sheet",
  csv: "sheet",
  tsv: "sheet",
  pptx: "slides",
};

export function documentKind(name: string, contentType = ""): DocumentKind | null {
  const dot = name.lastIndexOf(".");
  const byExtension = dot > 0 ? EXTENSION_KINDS[name.slice(dot + 1).toLowerCase()] : undefined;
  if (byExtension) return byExtension;
  return contentType.split(";", 1)[0]!.trim().toLowerCase() === "application/pdf" ? "pdf" : null;
}

/** Documents past this are offered as a download instead: each viewer parses on the main thread. */
export const MAX_DOCUMENT_PREVIEW_BYTES = 50 * 1024 * 1024;
