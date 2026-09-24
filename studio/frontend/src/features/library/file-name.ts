// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept free of app imports, so the naming rules can be tested on their own.
import { isTextAttachmentName } from "../chat/text-attachment-accept.ts";

// Characters no Windows file name may hold. "/" also ends a name for the desktop app's save
// dialog, which keeps only what follows it.
const INVALID_CHARS = new Set('<>:"/\\|?*');
// Device names Windows reserves whatever the extension: CON.txt opens the console.
const RESERVED_STEM = /^(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³]|conin\$|conout\$)$/i;
// Well inside every file system's 255, with room for a "(1)" the browser may add.
const MAX_NAME_CHARS = 200;
const FALLBACK_STEM = "file";

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

function clean(name: string): string {
  let out = "";
  for (const char of name) {
    const code = char.codePointAt(0)!;
    // Controls too, which no file manager shows usefully.
    out += INVALID_CHARS.has(char) || code < 0x20 || code === 0x7f ? "_" : char;
  }
  // Windows drops trailing dots and spaces, so "notes." and "notes" would collide.
  return out.replace(/[. ]+$/, "").trim();
}

/**
 * A name the item's file can be saved or attached under on any OS. The shown name may have been
 * renamed to anything, so it is cleaned, and gets back the extension of the file it names, which
 * is what says what the bytes are. Text-only chat uploads say so with a .txt.
 */
export function libraryFileName(item: {
  name: string;
  fileName?: string;
  textOnly?: boolean;
}): string {
  const extension = item.textOnly ? "txt" : extensionOf(item.fileName ?? item.name);
  const cleanExtension = clean(extension);
  let name = clean(item.name) || FALLBACK_STEM;
  if (cleanExtension && extensionOf(name) !== cleanExtension) name = `${name}.${cleanExtension}`;
  const dot = name.lastIndexOf(".");
  // A leading dot (".env") starts a name, not an extension.
  let stem = dot > 0 ? name.slice(0, dot) : name;
  const suffix = dot > 0 ? name.slice(dot) : "";
  if (!stem.trim()) stem = FALLBACK_STEM;
  // The device check reads up to the first dot: "con.backup.txt" is reserved too.
  const device = stem.split(".", 1)[0]!.trim();
  if (RESERVED_STEM.test(device)) stem = `_${stem}`;
  if (stem.length + suffix.length > MAX_NAME_CHARS) {
    stem = clean(stem.slice(0, Math.max(1, MAX_NAME_CHARS - suffix.length)));
  }
  return `${stem}${suffix}`;
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

const OPAQUE_TYPE = "application/octet-stream";

/**
 * The type a File under `fileName` should carry. The server's own type wins when it says anything;
 * otherwise (a Windows server knows no type for .py) it comes from the extension, and a name the
 * composer reads as text is plain text.
 */
export function libraryFileType(fileName: string, serverType: string): string {
  const type = serverType.split(";", 1)[0]!.trim().toLowerCase();
  if (type && type !== OPAQUE_TYPE) return type;
  const byExtension = EXTENSION_TYPES[extensionOf(fileName)];
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

/**
 * The type for an object URL a preview embeds directly. A PDF frame is always a PDF, and nothing
 * embedded is ever typed as a document that could run script: a web page previews only in the
 * sandboxed canvas frame. Media the server typed otherwise is left for the element to sniff.
 */
export function embeddedBlobType(body: EmbeddedBody, serverType: string): string {
  if (body === "pdf") return "application/pdf";
  const type = serverType.split(";", 1)[0]!.trim().toLowerCase();
  if (type.startsWith(`${body}/`) && !SCRIPTABLE_TYPES.has(type)) return type;
  return OPAQUE_TYPE;
}
