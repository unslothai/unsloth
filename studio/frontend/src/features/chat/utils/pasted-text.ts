// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Long pastes become a .txt attachment instead of flooding the composer.
// TextAttachmentAdapter still sends the full text to the model.

const PASTED_TEXT_MIME = "text/plain";
export const PASTED_TEXT_MIN_CHARS = 2000;
export const PASTED_TEXT_MIN_LINES = 40;
const PASTED_TEXT_NAME_RE = /^Pasted_Text_\d+\.txt$/;

type ClipboardTextPasteEvent = {
  readonly clipboardData: DataTransfer | null;
  readonly defaultPrevented: boolean;
  preventDefault: () => void;
};

// Identity separates a pasted blob from a .txt the user attached. Sent
// messages keep only the name, so isPastedTextFile falls back to that.
const pastedTextFiles = new WeakSet<File>();

function countLines(text: string): number {
  let lines = 1;
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === "\n") lines += 1;
  }
  return lines;
}

// No upper bound: the bigger the paste, the worse it is inline.
export function shouldAttachPastedText(text: string): boolean {
  if (text.trim().length === 0) return false;
  return (
    text.length >= PASTED_TEXT_MIN_CHARS ||
    countLines(text) >= PASTED_TEXT_MIN_LINES
  );
}

export function pastedTextFileName(now: number = Date.now()): string {
  return `Pasted_Text_${Math.floor(now / 1000)}.txt`;
}

export function createPastedTextFile(
  text: string,
  now: number = Date.now(),
): File {
  const file = new File([text], pastedTextFileName(now), {
    type: PASTED_TEXT_MIME,
    lastModified: now,
  });
  pastedTextFiles.add(file);
  return file;
}

export function isPastedTextFile(
  file: File | undefined,
  name?: string,
): boolean {
  // A live File must match by identity, or a .txt the user named this way
  // would be mistaken for a paste.
  if (file) return pastedTextFiles.has(file);
  return PASTED_TEXT_NAME_RE.test(name ?? "");
}

function clipboardHasFiles(clipboardData: DataTransfer): boolean {
  if (Array.from(clipboardData.files).some((file) => file.size > 0))
    return true;
  return Array.from(clipboardData.items).some((item) => item.kind === "file");
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

// Native (Tauri) images and copied files are advertised by type only, with no
// entry in files or items, and pasteClipboardFiles reads them itself.
function clipboardAdvertisesFiles(clipboardData: DataTransfer): boolean {
  const types = Array.from(clipboardData.types, (type) => type.toLowerCase());
  return (
    types.some((type) => type.startsWith("image/")) ||
    types.includes("files") ||
    types.some((type) => type.includes("copied-files")) ||
    clipboardHasLocalFileUri(clipboardData, types)
  );
}

function clipboardText(clipboardData: DataTransfer): string {
  try {
    return clipboardData.getData("text/plain");
  } catch {
    return "";
  }
}

/** Attaches an oversized text paste. True means the caller must not paste. */
export function pasteLongTextAsFile(
  event: ClipboardTextPasteEvent,
  addFile: (file: File) => void | Promise<void>,
  onError?: () => void,
): boolean {
  if (event.defaultPrevented) return false;
  const { clipboardData } = event;
  if (!clipboardData) return false;
  // Images and files keep the existing paste path.
  if (clipboardHasFiles(clipboardData)) return false;
  if (clipboardAdvertisesFiles(clipboardData)) return false;

  const text = clipboardText(clipboardData);
  if (!shouldAttachPastedText(text)) return false;

  event.preventDefault();
  void Promise.resolve(addFile(createPastedTextFile(text))).catch(() =>
    onError?.(),
  );
  return true;
}

const ATTACHMENT_WRAPPER_RE =
  /^<attachment name=[^\n>]*>\n([\s\S]*)\n<\/attachment>$/;

/** Strips the wrapper TextAttachmentAdapter adds on send, for previews. */
export function unwrapAttachmentText(text: string): string {
  return ATTACHMENT_WRAPPER_RE.exec(text)?.[1] ?? text;
}
