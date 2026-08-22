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
