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

// For a file picked without its extension.
const TYPE_KINDS: Record<string, DocumentKind> = {
  "application/pdf": "pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "sheet",
  "application/vnd.ms-excel.sheet.macroenabled.12": "sheet",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": "slides",
  "text/csv": "sheet",
  "text/tab-separated-values": "sheet",
};

// Own keys only: a name ending ".constructor" must not find Object.prototype's.
function lookUp(table: Record<string, DocumentKind>, key: string): DocumentKind | undefined {
  return Object.hasOwn(table, key) ? table[key] : undefined;
}

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

function mimeOf(contentType: string): string {
  return contentType.split(";", 1)[0]!.trim().toLowerCase();
}

export function documentKind(name: string, contentType = ""): DocumentKind | null {
  return lookUp(EXTENSION_KINDS, extensionOf(name)) ?? lookUp(TYPE_KINDS, mimeOf(contentType)) ?? null;
}

const DELIMITERS: Record<string, "," | "\t"> = {
  csv: ",",
  tsv: "\t",
  "text/csv": ",",
  "text/tab-separated-values": "\t",
};

/** A CSV or TSV's delimiter, by extension or else MIME type; null for any other file. */
export function sheetDelimiter(name: string, contentType = ""): "," | "\t" | null {
  const extension = extensionOf(name);
  const key = lookUp(EXTENSION_KINDS, extension) ? extension : mimeOf(contentType);
  return Object.hasOwn(DELIMITERS, key) ? DELIMITERS[key]! : null;
}

/** Documents past this are offered as a download instead: each viewer parses on the main thread. */
export const MAX_DOCUMENT_PREVIEW_BYTES = 50 * 1024 * 1024;
