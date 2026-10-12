// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const OPEN_DOCUMENT_SPREADSHEET_MIME =
  "application/vnd.oasis.opendocument.spreadsheet";
export const OPEN_DOCUMENT_TEXT_MIME =
  "application/vnd.oasis.opendocument.text";
export const OPEN_DOCUMENT_ATTACHMENT_EXTENSIONS = ".ods,.odt";
export const OPEN_DOCUMENT_ATTACHMENT_ACCEPT = [
  OPEN_DOCUMENT_ATTACHMENT_EXTENSIONS,
  OPEN_DOCUMENT_SPREADSHEET_MIME,
  OPEN_DOCUMENT_TEXT_MIME,
].join(",");

export function isOpenDocumentAttachmentName(filename: string): boolean {
  const lower = filename.toLowerCase();
  return OPEN_DOCUMENT_ATTACHMENT_EXTENSIONS.split(",").some((extension) =>
    lower.endsWith(extension),
  );
}

export const RTF_ATTACHMENT_EXTENSIONS = ".rtf";
export const RTF_MIMES = ["application/rtf", "text/rtf"];
export const RTF_ATTACHMENT_ACCEPT = [
  RTF_ATTACHMENT_EXTENSIONS,
  ...RTF_MIMES,
].join(",");

export function isRtfAttachmentName(filename: string): boolean {
  return filename.toLowerCase().endsWith(RTF_ATTACHMENT_EXTENSIONS);
}

// RAG_DOCUMENT_UPLOAD_ACCEPT minus what the browser reads itself (.xlsx/.xlsm/.pptx, .odt/.ods, .rtf,
// .eml as text). Spelled out: an import would stop this module being a leaf of the chat barrel.
export const SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS = [
  ".doc,.docm,.dotx,.dotm,.xls,.xltx,.xltm,.ppt,.pptm,.potx,.potm,.ppsx,.ppsm",
  ".odp,.ott,.ots,.otp,.msg,.mht,.mhtml,.epub,.xhtml,.xht",
].join(",");
// RAG's MAX_UPLOAD_BYTES; the browser readers stop at 50 MB, so larger documents go here too.
export const MAX_SERVER_DOCUMENT_BYTES = 200 * 1024 * 1024;

export function isServerDocumentAttachmentName(name: string): boolean {
  const lower = name.toLowerCase();
  return SERVER_DOCUMENT_ATTACHMENT_EXTENSIONS.split(",").some((ext) =>
    lower.endsWith(ext),
  );
}

// Matched by extension only: their MIME types are missing or shared with unrelated files.
// Keep in sync with TOOL_ONLY_ATTACHMENT_EXTS in native_path_policy.rs.
export const TOOL_ONLY_ATTACHMENT_EXTENSIONS = [
  ".parquet,.feather,.arrow,.orc,.dta,.sas7bdat,.xpt,.mat,.npy,.npz,.safetensors",
  ".sqlite,.sqlite3,.db,.gpkg,.mbtiles,.duckdb",
  ".zip,.jar,.whl,.apk,.tar,.gz,.tgz,.bz2,.tbz2,.tbz,.xz,.txz,.lzma",
  ".epub,.mobi,.fb2,.cbz,.xps,.oxps,.docm,.dotx,.dotm,.potx,.potm,.ppsm,.odp,.odg,.vsdx",
  ".stl,.3mf,.ply,.glb,.kmz,.ttf,.otf,.ttc,.woff",
  ".psd,.ico,.icns,.cur,.tga,.dds,.pcx,.ppm,.pgm,.pbm,.pnm,.qoi,.jp2,.j2k,.xbm,.xpm,.sgi,.fits",
].join(",");

export function isToolOnlyAttachmentName(name: string): boolean {
  const lower = name.toLowerCase();
  return TOOL_ONLY_ATTACHMENT_EXTENSIONS.split(",").some((ext) =>
    lower.endsWith(ext),
  );
}
