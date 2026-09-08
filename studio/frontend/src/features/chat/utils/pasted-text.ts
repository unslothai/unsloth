// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Long pastes become a .txt attachment instead of flooding the composer.
// TextAttachmentAdapter still sends the full text to the model.

import {
  clipboardAdvertisesFiles,
  clipboardHasFileEntries,
} from "./clipboard-payload.ts";

const PASTED_TEXT_MIME = "text/plain";
// Threshold the user picks in Settings > Chat. 0 keeps every paste inline.
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
// Enough for the name after whitespace collapses, without copying a line.
const PASTED_TEXT_NAME_SCAN_CHARS = 256;
// Blank lines are stepped over one at a time, so the walk needs its own bound.
const PASTED_TEXT_NAME_TOTAL_SCAN_CHARS = 64 * 1024;
const PASTED_TEXT_FALLBACK_NAME = "Pasted text";
const PASTED_TEXT_TAG = "pasted_text";
// A preview is for reading, so render an opening rather than megabytes.
export const PASTED_TEXT_PREVIEW_MAX_CHARS = 100_000;
// A title only ever uses the first line of this.
const ATTACHMENT_SAMPLE_CHARS = 512;
// Illegal in filenames on at least one platform, plus control characters.
const UNSAFE_NAME_CHARS = /[\\/:*?"<>|\p{Cc}]/gu;

type ClipboardTextPasteEvent = {
  readonly clipboardData: DataTransfer | null;
  readonly defaultPrevented: boolean;
  preventDefault: () => void;
};

type PlainPasteKeyEvent = {
  readonly code?: string;
  readonly key?: string;
  /** Deprecated, but layout-independent, and every engine still sets it. */
  readonly keyCode?: number;
  readonly metaKey: boolean;
  readonly ctrlKey: boolean;
  readonly shiftKey: boolean;
  readonly altKey: boolean;
};

/** `KeyboardEvent.keyCode` for V, which follows the physical key, not the character the layout and Option produce. */
const V_KEY_CODE = 86;

// Resolved once: the chord below is read on every keydown in the composer, and the platform
// cannot change under a live document.
let macPlatform: boolean | null = null;

function isMacPlatform(): boolean {
  if (macPlatform !== null) return macPlatform;
  if (typeof navigator === "undefined") return false;
  macPlatform = /mac|iphone|ipad|ipod/i.test(
    `${navigator.platform ?? ""} ${navigator.userAgent ?? ""}`,
  );
  return macPlatform;
}

/** The paste-without-formatting chord: Opt-Shift-Cmd-V on macOS, which is what its Edit menu
 *  carries and so the only one that actually pastes there, and Ctrl+Shift+V elsewhere. It
 *  asks for the clipboard in the field, so a paste made with it stays inline however long. */
export function isPlainPasteChord(
  event: PlainPasteKeyEvent,
  mac: boolean = isMacPlatform(),
): boolean {
  if (!event.shiftKey) return false;
  // The other platform's modifier is a different chord, not this one.
  if (mac ? !event.metaKey || event.ctrlKey : !event.ctrlKey || event.metaKey) {
    return false;
  }
  // Option belongs to the chord on macOS and to nothing off it. Shift-Cmd-V is taken on macOS
  // too: web apps bind it, so a host that maps it should reach the field.
  if (event.altKey && !mac) return false;
  // The layout decides where paste lives, not the board: Dvorak puts V on the QWERTY period key
  // and moves the chord with it. So a key that types a letter answers for itself, and a
  // letter that is not v is not this chord however it is wired.
  const typed = (event.key ?? "").toLowerCase();
  if (!event.altKey && typed.length === 1 && typed >= "a" && typed <= "z") {
    return typed === "v";
  }
  // No letter to read: Option rewrote it, or the layout types none. The two signals left
  // disagree on a remapped board -- `code` is the position, `keyCode` the letter -- so either
  // saying V is enough. That is safe because this only counts while the keys are still down,
  // and the one paste that can land by then is the chord's own.
  return event.code === "KeyV" || event.keyCode === V_KEY_CODE;
}

/** How long a plain-paste chord stands for the paste it asks for. The browser dispatches that
 *  paste while still handling the keydown, so the window need only survive one task. It must
 *  expire though: on macOS Shift-Cmd-V is a convention rather than a menu command, so it can
 *  paste nothing, and the Edit-menu paste reached for next has no keydown to clear it. */
export const PLAIN_PASTE_GESTURE_MS = 1000;

/** True when the paste being handled is the one that chord asked for. */
export function plainPasteStillCounts(chordAt: number, now: number): boolean {
  return chordAt > 0 && now - chordAt < PLAIN_PASTE_GESTURE_MS;
}

// Identity separates a pasted blob from a .txt the user attached. A sent message keeps no
// File, so the wrapper below carries it over instead.
const pastedTextFiles = new WeakSet<File>();
// The text as pasted, so draft autosave never re-reads the File on a keystroke.
const pastedTextByFile = new WeakMap<File, string>();

// Length alone decides. Whitespace is not exempt: the bigger the paste, the worse it is inline.
export function shouldAttachPastedText(
  text: string,
  minChars: number = PASTED_TEXT_DEFAULT_MIN_CHARS,
): boolean {
  if (text.length === 0) return false;
  if (minChars <= PASTED_TEXT_THRESHOLD_OFF) return false;
  return text.length >= minChars;
}

// Splitting a multi-megabyte paste to read one line allocates every other line for nothing, so
// walk to the first non-blank line instead, and never copy more than a name can hold.
function firstTextLine(text: string): string {
  // Bounded twice over: each slice, so one line cannot be copied whole, and the walk itself, so
  // a paste of nothing but blank lines cannot be stepped through a line at a time.
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

/** Names the file after the opening of the paste, so the chip is readable. */
export function pastedTextFileName(text: string): string {
  const cleaned = firstTextLine(text)
    .replace(UNSAFE_NAME_CHARS, " ")
    .replace(/\s+/g, " ")
    .trim();

  let snippet = cleaned.slice(0, PASTED_TEXT_NAME_MAX_CHARS);
  if (cleaned.length > PASTED_TEXT_NAME_MAX_CHARS) {
    // Prefer a word boundary, but not one that leaves a stub.
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

/** A live File must match by identity: the name no longer marks a paste. */
export function isPastedTextFile(file: File | undefined): boolean {
  return file !== undefined && pastedTextFiles.has(file);
}

/** What was pasted, without a `File.text()` round trip. */
export function pastedTextOf(file: File | undefined): string | undefined {
  return file === undefined ? undefined : pastedTextByFile.get(file);
}

/** Wraps an attachment for the model. A paste gets its own tag and its size, the only markers
 *  that survive a reload, since the File does not. */
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

// Header only. Sizing a sent paste any other way means walking every byte of it during render,
// which is what the chip must never do.
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

// Where the body sits inside the wrapper, so callers can slice out the piece they need instead
// of unwrapping megabytes. Unwrapped content is all body.
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

/** Previews wrapped content without unwrapping it: the body can be megabytes, and only the opening is ever rendered. */
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

/** The opening of an attachment's body, for naming a thread. A paste-only message has no
 *  inline text, so this is all the title has to work from. */
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

/** The same opening, taken from the first attachment that carries text. */
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

/** A pasted body without its wrapper, for anything showing the text as the user pasted it:
 *  copy, exports, fine-tuning rows. Other content is untouched. */
export function unwrapPastedTextContent(content: string): string {
  if (!isPastedTextContent(content)) return content;
  const { start, end } = attachmentBodyRange(content);
  return content.slice(start, end);
}

/** The whole body of a sent paste, unwrapped. Empty for any other content. */
export function pastedTextContentBody(content: string): string {
  return isPastedTextContent(content) ? unwrapPastedTextContent(content) : "";
}

/** Every pasted body on a message, for the paths that read `content` alone and would otherwise
 *  lose the paste: Chat Search and copying a message. Other attachment types are not
 *  included, since neither path ever had them. */
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
  // Images and files keep the existing paste path.
  if (clipboardHasFileEntries(clipboardData)) return false;
  if (clipboardAdvertisesFiles(clipboardData)) return false;

  const text = clipboardText(clipboardData);
  if (!shouldAttachPastedText(text, minChars)) return false;

  // Build the file first: if that throws, the browser can still paste.
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
    // addFile can throw before it ever returns a promise to catch on.
    onError?.();
  }
  return true;
}

/** Renders an opening rather than the whole paste, which can be megabytes. */
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
