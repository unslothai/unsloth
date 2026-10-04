// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Shared clipboard inspection. The file paste path and the long text paste path must agree on what
// counts as a file, or a paste falls through both or neither.

function clipboardTypes(clipboardData: DataTransfer): string[] {
  return Array.from(clipboardData.types, (type) => type.toLowerCase());
}

function clipboardHasLocalFileUri(
  clipboardData: DataTransfer,
  types: readonly string[],
): boolean {
  return types
    .filter((type) => type.includes("uri-list") || type.includes("urilist"))
    .some((type) => {
      try {
        return clipboardData
          .getData(type)
          .split(/\r?\n/)
          .some((line) => line.trim().toLowerCase().startsWith("file:"));
      } catch {
        return false;
      }
    });
}

/** Files the browser decoded for us, ready to attach. */
export function browserClipboardFiles(clipboardData: DataTransfer): File[] {
  const files = Array.from(clipboardData.files).filter((file) => file.size > 0);
  if (files.length > 0) return files;

  return Array.from(clipboardData.items)
    .filter((item) => item.kind === "file")
    .map((item) => item.getAsFile())
    .filter((file): file is File => file !== null && file.size > 0);
}

/** A file entry the browser may still refuse to decode, so the text path defers. */
export function clipboardHasFileEntries(clipboardData: DataTransfer): boolean {
  if (Array.from(clipboardData.files).some((file) => file.size > 0)) return true;
  return Array.from(clipboardData.items).some((item) => item.kind === "file");
}

// Native (Tauri) images and copied files are advertised by type only, with no entry in files or
// items, and only pasteClipboardFiles can read them.
export function clipboardAdvertisesFiles(clipboardData: DataTransfer): boolean {
  const types = clipboardTypes(clipboardData);
  return (
    types.some((type) => type.startsWith("image/")) ||
    types.includes("files") ||
    types.some((type) => type.includes("copied-files")) ||
    clipboardHasLocalFileUri(clipboardData, types)
  );
}

export function clipboardHasPlainText(clipboardData: DataTransfer): boolean {
  try {
    return clipboardData.getData("text/plain").length > 0;
  } catch {
    return true;
  }
}
