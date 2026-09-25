// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How a text file reads, and how an edit goes back in the same shape. Kept free of app imports, so
// it can be tested on its own.

export type NoteEncoding = "utf-8" | "utf-16le" | "utf-16be";

export interface NoteFormat {
  encoding: NoteEncoding;
  /** The file started with a byte order mark, which a save must keep. */
  bom: boolean;
  /** The file's line ending: text on screen always uses "\n". */
  eol: "\n" | "\r\n";
}

export interface DecodedNote {
  /** With "\n" line endings, as a textarea reports its value. */
  text: string;
  format: NoteFormat;
  /** Why the file opens read-only, or null when it can be edited and saved back as it was. */
  readOnlyReason: NoteReadOnlyReason | null;
}

/** Saves go back in the file's own encoding, so only text that decodes cleanly is editable. The
 * preview words these. */
export type NoteReadOnlyReason = "utf16" | "notUtf8";

const UTF16_READ_ONLY: NoteReadOnlyReason = "utf16";
const NOT_UTF8_READ_ONLY: NoteReadOnlyReason = "notUtf8";

function detectEncoding(bytes: Uint8Array): { encoding: NoteEncoding; bom: boolean } {
  if (bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) {
    return { encoding: "utf-8", bom: true };
  }
  if (bytes[0] === 0xff && bytes[1] === 0xfe) return { encoding: "utf-16le", bom: true };
  if (bytes[0] === 0xfe && bytes[1] === 0xff) return { encoding: "utf-16be", bom: true };
  return { encoding: "utf-8", bom: false };
}

/** The ending most of the file's lines use; one with no line breaks gets "\n". */
function detectEol(text: string): "\n" | "\r\n" {
  const crlf = text.match(/\r\n/g)?.length ?? 0;
  const lf = (text.match(/\n/g)?.length ?? 0) - crlf;
  return crlf > lf ? "\r\n" : "\n";
}

/**
 * Decode a file's bytes, or the first of them when `truncated`, where a character cut at the end is
 * dropped rather than read as corrupt. The BOM decides the encoding; with none the file must be
 * valid UTF-8 to be editable, since saving a legacy code page as UTF-8 would rewrite every accented
 * letter in it.
 */
export function decodeNote(bytes: Uint8Array, truncated = false): DecodedNote {
  const { encoding, bom } = detectEncoding(bytes);
  let raw: string;
  let readOnlyReason: NoteReadOnlyReason | null = null;
  try {
    // The decoder drops the BOM itself.
    raw = new TextDecoder(encoding, { fatal: true }).decode(bytes, { stream: truncated });
  } catch {
    raw = new TextDecoder(encoding).decode(bytes, { stream: truncated });
    readOnlyReason = encoding === "utf-8" ? NOT_UTF8_READ_ONLY : UTF16_READ_ONLY;
  }
  return {
    text: raw.replace(/\r\n/g, "\n"),
    format: { encoding, bom, eol: detectEol(raw) },
    readOnlyReason,
  };
}

/** The text to write for an edit of a file read as `format`: its line endings and BOM restored. */
export function encodeNote(text: string, format: NoteFormat): string {
  // A textarea reports "\n" alone; "\r\n" typed or pasted in collapses first so it is not doubled.
  const lines = text.replace(/\r\n/g, "\n");
  const body = format.eol === "\r\n" ? lines.replace(/\n/g, "\r\n") : lines;
  return format.bom ? `\uFEFF${body}` : body;
}
