// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How a Library file is named, typed and served. Kept free of app imports, so it can be tested.
import { isTextAttachmentName } from "../chat/text-attachment-accept.ts";

// Characters no Windows file name may hold; "/" also cuts the desktop save dialog's name short.
const INVALID_CHARS = new Set('<>:"/\\|?*');
// Device names Windows reserves whatever the extension: CON.txt opens the console.
const RESERVED_STEM = /^(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³]|conin\$|conout\$)$/i;
// Well inside every file system's 255 (UTF-8 bytes on Linux and macOS, UTF-16 units on Windows),
// with room for a "(1)" the browser may add.
const MAX_NAME_BYTES = 200;
const utf8 = new TextEncoder();
const FALLBACK_STEM = "file";
const OPAQUE_TYPE = "application/octet-stream";

export function fileExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

/** Stem and ".ext"; a leading dot (".env") starts a name, not an extension. */
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
  // Windows drops trailing dots and spaces, so "notes." and "notes" would collide.
  return out.replace(/[. ]+$/, "").trim();
}

/** A name the item's file can be saved or attached under on any OS: the shown name, cleaned, with
 *  the extension of the file it names (which a rename may have dropped). Text-only uploads get .txt. */
export function libraryFileName(item: {
  name: string;
  fileName?: string;
  textOnly?: boolean;
}): string {
  const extension = clean(item.textOnly ? "txt" : fileExtension(item.fileName ?? item.name));
  let name = clean(item.name) || FALLBACK_STEM;
  if (extension && fileExtension(name) !== extension) name = `${name}.${extension}`;
  const [base, suffix] = splitName(name);
  let stem = base.trim() ? base : FALLBACK_STEM;
  // The device check reads up to the first dot: "con.backup.txt" is reserved too.
  if (RESERVED_STEM.test(stem.split(".", 1)[0]!.trim())) stem = `_${stem}`;
  const budget = MAX_NAME_BYTES - utf8.encode(suffix).length;
  if (utf8.encode(stem).length > budget) {
    // By whole characters, so none is cut in half; at least the first is kept.
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

/** The names, with " (2)", " (3)"... before the extension of any already taken (ignoring case, as
 *  Windows and macOS do), so a folder of downloads keeps every file. */
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

// Types for what the composer and the browser read by type; the rest go by extension.
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

/** "text/x-python; charset=utf-8" as "text/x-python". */
function baseType(type: string): string {
  return type.split(";", 1)[0]!.trim().toLowerCase();
}

/** The type a File under `fileName` should carry: the server's, unless it says nothing (a Windows
 *  server knows no type for .py); then the extension's, or plain text for what the composer reads. */
export function libraryFileType(fileName: string, serverType: string): string {
  const type = baseType(serverType);
  if (type && type !== OPAQUE_TYPE) return type;
  const byExtension = EXTENSION_TYPES[fileExtension(fileName)];
  if (byExtension) return byExtension;
  return isTextAttachmentName(fileName) ? "text/plain" : OPAQUE_TYPE;
}

// Types a browser runs script in when it loads them as a document.
const SCRIPTABLE_TYPES = new Set([
  "text/html",
  "application/xhtml+xml",
  "image/svg+xml",
  "text/xml",
  "application/xml",
]);

export type EmbeddedBody = "image" | "pdf" | "audio" | "video";

/** The type for an object URL a preview embeds: a PDF frame is always a PDF, and nothing is ever
 *  typed as a document that could run script (web pages preview only in the sandboxed canvas
 *  frame). Media the server typed otherwise is left for the element to sniff. */
export function embeddedBlobType(body: EmbeddedBody, serverType: string): string {
  if (body === "pdf") return "application/pdf";
  const type = baseType(serverType);
  return type.startsWith(`${body}/`) && !SCRIPTABLE_TYPES.has(type) ? type : OPAQUE_TYPE;
}

/** Which version of an item's file a cache holds: its time, and its size, since a chat attachment
 *  rewritten in place keeps its message's time. */
export function itemVersion(item: { id: string; updatedAt: number; sizeBytes: number | null }): string {
  return `${item.id}@${item.updatedAt}.${item.sizeBytes ?? ""}`;
}

/** Items with a file of their own, which the Library serves by id. Chat attachments live inside
 *  their messages, and fine-tunes are folders. */
export function hasOwnFile(itemId: string): boolean {
  return /^(upload|image|video|audio|sandbox):/.test(itemId);
}

// Sources the backend streams from a signed link.
const STREAMED_SOURCES = new Set(["upload", "audio", "video", "sandbox"]);

/** Whether a preview plays from a signed, range-capable link rather than a buffered blob. */
export function streamsPreview(itemId: string, body: string | null): boolean {
  if (body !== "audio" && body !== "video") return false;
  const colon = itemId.indexOf(":");
  return colon > 0 && STREAMED_SOURCES.has(itemId.slice(0, colon));
}
