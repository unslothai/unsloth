// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  clipboardAdvertisesFiles,
  clipboardHasFileEntries,
} from "./clipboard-payload.ts";

const PASTED_TEXT_MIME = "text/plain";
// 0 keeps every paste inline.
export const PASTED_TEXT_THRESHOLD_OFF = 0;
export const PASTED_TEXT_DEFAULT_MIN_CHARS = 4000;
export const PASTED_TEXT_THRESHOLD_CHOICES = [
  PASTED_TEXT_THRESHOLD_OFF,
  2000,
  PASTED_TEXT_DEFAULT_MIN_CHARS,
  8000,
  16000,
] as const;
const PASTED_TEXT_NAME_MAX_CHARS = 32;
const PASTED_TEXT_NAME_SCAN_CHARS = 256;
const PASTED_TEXT_NAME_TOTAL_SCAN_CHARS = 64 * 1024;
const PASTED_TEXT_FALLBACK_NAME = "Pasted text";
const PASTED_TEXT_TAG = "pasted_text";
export const PASTED_TEXT_PREVIEW_MAX_CHARS = 100_000;
const ATTACHMENT_SAMPLE_CHARS = 512;
const UNSAFE_NAME_CHARS = /[\\/:*?"<>|\p{Cc}]/gu;

type ClipboardTextPasteEvent = {
  readonly clipboardData: DataTransfer | null;
  readonly defaultPrevented: boolean;
  preventDefault: () => void;
};

type PlainPasteKeyEvent = {
  readonly code?: string;
  readonly key?: string;
  readonly keyCode?: number;
  readonly metaKey: boolean;
  readonly ctrlKey: boolean;
  readonly shiftKey: boolean;
  readonly altKey: boolean;
};

/** Follows the physical key, not the layout or Option-produced character. */
const V_KEY_CODE = 86;

let macPlatform: boolean | null = null;

function isMacPlatform(): boolean {
  if (macPlatform !== null) return macPlatform;
  if (typeof navigator === "undefined") return false;
  macPlatform = /mac|iphone|ipad|ipod/i.test(
    `${navigator.platform ?? ""} ${navigator.userAgent ?? ""}`,
  );
  return macPlatform;
}

/** Opt-Shift-Cmd-V on macOS (its Edit menu chord), Ctrl+Shift+V elsewhere. */
export function isPlainPasteChord(
  event: PlainPasteKeyEvent,
  mac: boolean = isMacPlatform(),
): boolean {
  if (!event.shiftKey) return false;
  if (mac ? !event.metaKey || event.ctrlKey : !event.ctrlKey || event.metaKey) {
    return false;
  }
  if (event.altKey && !mac) return false;
  // The layout decides: a key that types a letter answers for itself (e.g. Dvorak).
  const typed = (event.key ?? "").toLowerCase();
  if (!event.altKey && typed.length === 1 && typed >= "a" && typed <= "z") {
    return typed === "v";
  }
  // No letter (Option rewrote it): either `code` or `keyCode` saying V is enough.
  return event.code === "KeyV" || event.keyCode === V_KEY_CODE;
}

/** Must expire: on macOS Shift-Cmd-V may paste nothing, and the next paste has no keydown. */
export const PLAIN_PASTE_GESTURE_MS = 1000;

export function plainPasteStillCounts(chordAt: number, now: number): boolean {
  return chordAt > 0 && now - chordAt < PLAIN_PASTE_GESTURE_MS;
}

// Identity marks a pasted blob; a sent message keeps no File, so the wrapper carries it.
const pastedTextFiles = new WeakSet<File>();
const pastedTextByFile = new WeakMap<File, string>();

export function shouldAttachPastedText(
  text: string,
  minChars: number = PASTED_TEXT_DEFAULT_MIN_CHARS,
): boolean {
  if (text.length === 0) return false;
  if (minChars <= PASTED_TEXT_THRESHOLD_OFF) return false;
  return text.length >= minChars;
}

function firstTextLine(text: string): string {
  const limit = Math.min(text.length, PASTED_TEXT_NAME_TOTAL_SCAN_CHARS);
  let start = 0;
  while (start < limit) {
    const end = text.indexOf("\n", start);
    const stop = Math.min(
      end === -1 ? text.length : end,
      start + PASTED_TEXT_NAME_SCAN_CHARS,
    );
    const line = text.slice(start, stop);
    if (line.trim().length > 0) return line;
    if (end === -1) return "";
    start = end + 1;
  }
  return "";
}

export function pastedTextFileName(text: string): string {
  const cleaned = firstTextLine(text)
    .replace(UNSAFE_NAME_CHARS, " ")
    .replace(/\s+/g, " ")
    .trim();

  let snippet = cleaned.slice(0, PASTED_TEXT_NAME_MAX_CHARS);
  if (cleaned.length > PASTED_TEXT_NAME_MAX_CHARS) {
    const lastSpace = snippet.lastIndexOf(" ");
    if (lastSpace >= PASTED_TEXT_NAME_MAX_CHARS / 2) {
      snippet = snippet.slice(0, lastSpace);
    }
  }
  snippet = snippet.replace(/[\s.]+$/, "");
  return `${snippet.length > 0 ? snippet : PASTED_TEXT_FALLBACK_NAME}.txt`;
}

export function createPastedTextFile(text: string): File {
  const file = new File([text], pastedTextFileName(text), {
    type: PASTED_TEXT_MIME,
    lastModified: Date.now(),
  });
  pastedTextFiles.add(file);
  pastedTextByFile.set(file, text);
  return file;
}

export function isPastedTextFile(file: File | undefined): boolean {
  return file !== undefined && pastedTextFiles.has(file);
}

export function pastedTextOf(file: File | undefined): string | undefined {
  return file === undefined ? undefined : pastedTextByFile.get(file);
}

/** The tag and size are the only paste markers that survive a reload. */
export function attachmentContentText(
  name: string,
  text: string,
  pasted: boolean,
  bytes?: number,
): string {
  if (!pasted) return `<attachment name=${name}>\n${text}\n</attachment>`;
  const size = bytes === undefined ? "" : ` bytes=${bytes}`;
  return `<${PASTED_TEXT_TAG} name=${name}${size}>\n${text}\n</${PASTED_TEXT_TAG}>`;
}

export function isPastedTextContent(text: string | undefined): boolean {
  return text?.startsWith(`<${PASTED_TEXT_TAG} name=`) === true;
}

// Header only, so the chip never walks every byte during render.
const PASTED_TEXT_BYTES_RE = /^<pasted_text name=[^\n>]* bytes=(\d+)>/;
const PASTED_TEXT_HEADER_SCAN_CHARS = 1024;

export function pastedTextContentBytes(
  content: string | undefined,
): number | undefined {
  if (content === undefined) return undefined;
  const bytes = PASTED_TEXT_BYTES_RE.exec(
    content.slice(0, PASTED_TEXT_HEADER_SCAN_CHARS),
  )?.[1];
  return bytes === undefined ? undefined : Number(bytes);
}

function attachmentBodyRange(content: string): { start: number; end: number } {
  const tag = content.startsWith("<attachment name=")
    ? "attachment"
    : isPastedTextContent(content)
      ? PASTED_TEXT_TAG
      : undefined;
  const headerEnd = tag === undefined ? -1 : content.indexOf("\n");
  if (headerEnd === -1) return { start: 0, end: content.length };

  const closing = `\n</${tag}>`;
  return {
    start: headerEnd + 1,
    end: content.endsWith(closing)
      ? Math.max(content.length - closing.length, headerEnd + 1)
      : content.length,
  };
}

export function pastedTextContentPreview(content: string): {
  text: string;
  remaining: number;
} {
  const { start, end } = attachmentBodyRange(content);
  const length = end - start;
  const taken = Math.min(length, PASTED_TEXT_PREVIEW_MAX_CHARS);
  return {
    text: content.slice(start, start + taken),
    remaining: length - taken,
  };
}

export function attachmentContentSample(
  content: string,
  max: number = ATTACHMENT_SAMPLE_CHARS,
): string {
  const { start, end } = attachmentBodyRange(content);
  return content.slice(start, Math.min(end, start + max)).trim();
}

type AttachmentLike = {
  readonly content?: readonly {
    readonly type: string;
    readonly text?: string;
  }[];
};

export function attachmentsSample(
  attachments: readonly AttachmentLike[] | undefined,
): string {
  for (const attachment of attachments ?? []) {
    for (const part of attachment.content ?? []) {
      if (part.type !== "text" || part.text === undefined) continue;
      const sample = attachmentContentSample(part.text);
      if (sample.length > 0) return sample;
    }
  }
  return "";
}

export function unwrapPastedTextContent(content: string): string {
  if (!isPastedTextContent(content)) return content;
  const { start, end } = attachmentBodyRange(content);
  return content.slice(start, end);
}

export function pastedTextContentBody(content: string): string {
  return isPastedTextContent(content) ? unwrapPastedTextContent(content) : "";
}

export function attachmentsPastedText(
  attachments: readonly AttachmentLike[] | undefined,
): string {
  const bodies: string[] = [];
  for (const attachment of attachments ?? []) {
    for (const part of attachment.content ?? []) {
      if (part.type !== "text" || part.text === undefined) continue;
      const body = pastedTextContentBody(part.text);
      if (body.length > 0) bodies.push(body);
    }
  }
  return bodies.join("\n\n");
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
  minChars?: number,
): boolean {
  if (event.defaultPrevented) return false;
  const { clipboardData } = event;
  if (!clipboardData) return false;
  if (clipboardHasFileEntries(clipboardData)) return false;
  if (clipboardAdvertisesFiles(clipboardData)) return false;

  const text = clipboardText(clipboardData);
  if (!shouldAttachPastedText(text, minChars)) return false;

  let file: File;
  try {
    file = createPastedTextFile(text);
  } catch {
    return false;
  }

  event.preventDefault();
  try {
    void Promise.resolve(addFile(file)).catch(() => onError?.());
  } catch {
    // addFile can throw synchronously before returning a promise.
    onError?.();
  }
  return true;
}

export function pastedTextPreview(text: string): {
  text: string;
  remaining: number;
} {
  const remaining = Math.max(text.length - PASTED_TEXT_PREVIEW_MAX_CHARS, 0);
  return {
    text: remaining > 0 ? text.slice(0, PASTED_TEXT_PREVIEW_MAX_CHARS) : text,
    remaining,
  };
}
