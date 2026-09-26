// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTextAttachmentName } from "../chat/text-attachment-accept.ts";

const INVALID_CHARS = new Set('<>:"/\\|?*');
const RESERVED_STEM = /^(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³]|conin\$|conout\$)$/i;
const MAX_NAME_BYTES = 200;
const utf8 = new TextEncoder();
const FALLBACK_STEM = "file";
const OPAQUE_TYPE = "application/octet-stream";

export function fileExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

function splitName(name: string): [stem: string, suffix: string] {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? [name.slice(0, dot), name.slice(dot)] : [name, ""];
}

function clean(name: string): string {
  let out = "";
  for (const char of name) {
    const code = char.codePointAt(0)!;
    out += INVALID_CHARS.has(char) || code < 0x20 || code === 0x7f ? "_" : char;
  }
  return out.replace(/[. ]+$/, "").trim();
}

export function libraryFileName(item: {
  name: string;
  fileName?: string;
  textOnly?: boolean;
}): string {
  // A text file chat stores whole (CSV, markdown, code) keeps its extension; extracted text is .txt.
  const own = item.fileName ?? item.name;
  const extension = clean(item.textOnly && !isTextAttachmentName(own) ? "txt" : fileExtension(own));
  let name = clean(item.name) || FALLBACK_STEM;
  if (extension && fileExtension(name) !== extension) name = `${name}.${extension}`;
  const [base, suffix] = splitName(name);
  let stem = base.trim() ? base : FALLBACK_STEM;
  if (RESERVED_STEM.test(stem.split(".", 1)[0]!.trim())) stem = `_${stem}`;
  const budget = MAX_NAME_BYTES - utf8.encode(suffix).length;
  if (utf8.encode(stem).length > budget) {
    let cut = "";
    let bytes = 0;
    for (const char of stem) {
      bytes += utf8.encode(char).length;
      if (bytes > budget && cut) break;
      cut += char;
    }
    stem = clean(cut) || FALLBACK_STEM;
  }
  return `${stem}${suffix}`;
}

export function uniqueFileNames(names: string[]): string[] {
  const taken = new Set<string>();
  return names.map((name) => {
    const [stem, suffix] = splitName(name);
    let candidate = name;
    for (let n = 2; taken.has(candidate.toLowerCase()); n++) candidate = `${stem} (${n})${suffix}`;
    taken.add(candidate.toLowerCase());
    return candidate;
  });
}

const EXTENSION_TYPES: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  avif: "image/avif",
  bmp: "image/bmp",
  svg: "image/svg+xml",
  heic: "image/heic",
  tif: "image/tiff",
  tiff: "image/tiff",
  pdf: "application/pdf",
  html: "text/html",
  htm: "text/html",
  md: "text/markdown",
  csv: "text/csv",
  tsv: "text/tab-separated-values",
  json: "application/json",
  mp3: "audio/mpeg",
  wav: "audio/wav",
  ogg: "audio/ogg",
  flac: "audio/flac",
  m4a: "audio/mp4",
  mp4: "video/mp4",
  mov: "video/quicktime",
  webm: "video/webm",
  mkv: "video/x-matroska",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  pptx: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
};

function baseType(type: string): string {
  return type.split(";", 1)[0]!.trim().toLowerCase();
}

export function libraryFileType(fileName: string, serverType: string): string {
  const type = baseType(serverType);
  if (type && type !== OPAQUE_TYPE) return type;
  const byExtension = EXTENSION_TYPES[fileExtension(fileName)];
  if (byExtension) return byExtension;
  return isTextAttachmentName(fileName) ? "text/plain" : OPAQUE_TYPE;
}

const SCRIPTABLE_TYPES = new Set([
  "text/html",
  "application/xhtml+xml",
  "image/svg+xml",
  "text/xml",
  "application/xml",
]);

export type EmbeddedBody = "image" | "audio" | "video";

export function embeddedBlobType(body: EmbeddedBody, serverType: string): string {
  const type = baseType(serverType);
  return type.startsWith(`${body}/`) && !SCRIPTABLE_TYPES.has(type) ? type : OPAQUE_TYPE;
}

export function itemVersion(item: { id: string; updatedAt: number; sizeBytes: number | null }): string {
  return `${item.id}@${item.updatedAt}.${item.sizeBytes ?? ""}`;
}

export function hasOwnFile(itemId: string): boolean {
  return /^(upload|image|video|audio|sandbox):/.test(itemId);
}

const STREAMED_SOURCES = new Set(["upload", "audio", "video", "sandbox"]);

export function streamsPreview(itemId: string, body: string | null): boolean {
  if (body !== "audio" && body !== "video") return false;
  const colon = itemId.indexOf(":");
  return colon > 0 && STREAMED_SOURCES.has(itemId.slice(0, colon));
}
