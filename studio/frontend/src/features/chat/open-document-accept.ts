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

export const OFFICE_OPEN_XML_MIMES = [
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.ms-excel.sheet.macroEnabled.12",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.template",
  "application/vnd.ms-excel.template.macroEnabled.12",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "application/vnd.ms-powerpoint.presentation.macroEnabled.12",
  "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
];
export const OFFICE_OPEN_XML_ATTACHMENT_EXTENSIONS =
  ".xlsx,.xlsm,.xltx,.xltm,.pptx,.pptm,.ppsx";
export const OFFICE_OPEN_XML_ATTACHMENT_ACCEPT = [
  OFFICE_OPEN_XML_ATTACHMENT_EXTENSIONS,
  ...OFFICE_OPEN_XML_MIMES,
].join(",");

export function isOfficeOpenXmlAttachmentName(filename: string): boolean {
  const lower = filename.toLowerCase();
  return OFFICE_OPEN_XML_ATTACHMENT_EXTENSIONS.split(",").some((extension) =>
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

export const IWORK_MIMES = [
  "application/vnd.apple.pages",
  "application/vnd.apple.numbers",
  "application/vnd.apple.keynote",
  "application/x-iwork-pages-sffpages",
  "application/x-iwork-numbers-sffnumbers",
  "application/x-iwork-keynote-sffkey",
];
export const IWORK_ATTACHMENT_EXTENSIONS = ".pages,.numbers,.key";
export const IWORK_ATTACHMENT_ACCEPT = [
  IWORK_ATTACHMENT_EXTENSIONS,
  ...IWORK_MIMES,
].join(",");

export function isIworkAttachmentName(filename: string): boolean {
  const lower = filename.toLowerCase();
  return IWORK_ATTACHMENT_EXTENSIONS.split(",").some((extension) =>
    lower.endsWith(extension),
  );
}

// Matched by extension only: their MIME types are missing or shared with unrelated files.
// Keep in sync with TOOL_ONLY_ATTACHMENT_EXTS in native_path_policy.rs.
export const TOOL_ONLY_ATTACHMENT_EXTENSIONS =
  ".parquet,.feather,.arrow,.orc,.sqlite,.sqlite3,.db,.zip,.tar,.gz,.tgz,.bz2,.xz,.npy,.npz,.epub,.mobi,.xps,.oxps,.docm,.dotx,.odp,.odg";

export function isToolOnlyAttachmentName(name: string): boolean {
  const lower = name.toLowerCase();
  return TOOL_ONLY_ATTACHMENT_EXTENSIONS.split(",").some((ext) =>
    lower.endsWith(ext),
  );
}
