// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Files that run code when opened always ask before downloading, whatever a site's remembered answer or
// the Ask before downloading setting.

import policy from "./dangerous-file-types.json" with { type: "json" };

const DANGEROUS = new Set<string>(policy.extensions);

// native_file_dialogs.rs MAX_DOWNLOAD_NAME_BYTES.
const MAX_NAME_BYTES = 240;
const utf8 = new TextEncoder();
const byteLength = (text: string) => utf8.encode(text).length;
// Saved with a leading "_" (native_file_dialogs.rs WINDOWS_DEVICE_NAMES).
const DEVICE = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;

/** The name the desktop app saves it under (native_file_dialogs.rs `safe_download_name`): past 240 bytes it
 *  keeps an extension of at most 32 bytes, else cuts the end, so a long tail can leave `.exe` last. */
function savedAs(name: string): string {
  const shift = DEVICE.test(name.split(".")[0].trimEnd()) ? 1 : 0;
  if (byteLength(name) + shift <= MAX_NAME_BYTES) return name;
  const dot = name.lastIndexOf(".");
  const extension = dot > 0 && byteLength(name.slice(dot)) <= 32 ? name.slice(dot) : "";
  const room = MAX_NAME_BYTES - shift - byteLength(extension);
  let stem = "";
  let used = 0;
  for (const char of name) {
    used += byteLength(char);
    if (used > room) break;
    stem += char;
  }
  return stem + extension;
}

function runsCode(name: string): boolean {
  const base = name.replace(/[. ]+$/, "");
  const dot = base.lastIndexOf(".");
  if (dot < 0) return false;
  // A bare `.exe` still runs as one. Windows compares names upper-cased, so `.m\u017fi` (long s) is `.MSI`.
  const extension = base.slice(dot + 1);
  return DANGEROUS.has(extension.toLowerCase()) || DANGEROUS.has(extension.toUpperCase().toLowerCase());
}

const PROGRAM_TYPES: Record<string, string> = {
  "application/x-msdownload": "exe",
  "application/x-msdos-program": "exe",
  "application/x-ms-dos-executable": "exe",
  "application/vnd.microsoft.portable-executable": "exe",
  "application/x-dosexec": "exe",
  "application/x-msi": "msi",
  "application/x-ms-installer": "msi",
  "application/x-apple-diskimage": "dmg",
  "application/x-sh": "sh",
  "application/x-shellscript": "sh",
  "text/x-sh": "sh",
  "application/java-archive": "jar",
  "application/x-java-archive": "jar",
  "application/vnd.debian.binary-package": "deb",
  "application/x-debian-package": "deb",
  "application/x-rpm": "rpm",
  "application/x-redhat-package-manager": "rpm",
  "application/hta": "hta",
  "text/javascript": "js",
  "application/javascript": "js",
  "application/x-javascript": "js",
};

const COMMON_TYPES: Record<string, string> = {
  "application/pdf": "pdf",
  "text/plain": "txt",
  "text/csv": "csv",
  "text/tab-separated-values": "tsv",
  "application/json": "json",
  "text/markdown": "md",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
  "image/avif": "avif",
  "audio/mpeg": "mp3",
  "audio/wav": "wav",
  "audio/ogg": "ogg",
  "audio/flac": "flac",
  "audio/mp4": "m4a",
  "video/mp4": "mp4",
  "video/webm": "webm",
  "video/quicktime": "mov",
  "application/zip": "zip",
  "application/gzip": "gz",
  "image/svg+xml": "svg",
  "image/bmp": "bmp",
  "image/tiff": "tiff",
  "image/x-icon": "ico",
  "image/vnd.microsoft.icon": "ico",
  "image/heic": "heic",
  "application/epub+zip": "epub",
  "application/xml": "xml",
  "text/xml": "xml",
  "application/rtf": "rtf",
  "text/rtf": "rtf",
  "text/calendar": "ics",
  "application/msword": "doc",
  "application/vnd.ms-excel": "xls",
  "application/vnd.ms-powerpoint": "ppt",
  "application/vnd.oasis.opendocument.text": "odt",
  "application/vnd.oasis.opendocument.spreadsheet": "ods",
  "application/vnd.oasis.opendocument.presentation": "odp",
  "application/x-7z-compressed": "7z",
  "application/x-tar": "tar",
  "application/vnd.rar": "rar",
  "application/x-rar-compressed": "rar",
  "application/x-bzip2": "bz2",
  "application/x-xz": "xz",
  "application/zstd": "zst",
  "audio/webm": "webm",
  "font/woff": "woff",
  "font/woff2": "woff2",
  "font/ttf": "ttf",
  "font/otf": "otf",
};

export function withTypeExtension(name: string, contentType: string, includeCommonTypes = false): string {
  const type = contentType.split(";")[0].trim().toLowerCase();
  const extension = PROGRAM_TYPES[type] ?? (includeCommonTypes ? COMMON_TYPES[type] : undefined);
  const stem = name.replace(/^\.+/, "").replace(/[. ]+$/, "");
  return extension && !/\.[^./\\]+$/.test(stem) ? `${stem || "download"}.${extension}` : name;
}

/** As the desktop app cleans a name before the cut: controls, bidi controls and reserved characters become
 *  one-byte "_", trailing dots and spaces go. */
function cleaned(name: string): string {
  return name
    .replace(/[\u0000-\u001f\u007f-\u009f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069/\\:*?"<>|]/g, "_")
    .replace(/[. ]+$/, "");
}

/** Whether `name` runs code when opened, judged as the OS will (Windows drops trailing dots and spaces),
 *  both as given and as the desktop app shortens it. Its last path part and the whole name (separators as
 *  "_") are both judged: which one the app saves depends on where the name came from. */
export function isDangerousDownload(name: string): boolean {
  const base = name.slice(Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\")) + 1);
  return runsCode(base) || [base, name].some((candidate) => runsCode(savedAs(cleaned(candidate))));
}
