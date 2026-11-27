// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export type NoteEncoding = "utf-8" | "utf-16le" | "utf-16be";

export interface NoteFormat {
  encoding: NoteEncoding;
  bom: boolean;
  eol: "\n" | "\r\n";
}

export type NoteReadOnlyReason = "utf16" | "notUtf8";

export interface DecodedNote {
  text: string;
  format: NoteFormat;
  readOnlyReason: NoteReadOnlyReason | null;
}

function detectEncoding(bytes: Uint8Array): { encoding: NoteEncoding; bom: boolean } {
  if (bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) {
    return { encoding: "utf-8", bom: true };
  }
  if (bytes[0] === 0xff && bytes[1] === 0xfe) return { encoding: "utf-16le", bom: true };
  if (bytes[0] === 0xfe && bytes[1] === 0xff) return { encoding: "utf-16be", bom: true };
  return { encoding: "utf-8", bom: false };
}

/**
 * The BOM decides the encoding; without one the file must be valid UTF-8 to be editable,
 * since saving a legacy code page as UTF-8 would rewrite accented letters.
 */
export function decodeNote(bytes: Uint8Array, truncated = false): DecodedNote {
  const { encoding, bom } = detectEncoding(bytes);
  let raw: string;
  let readOnlyReason: NoteReadOnlyReason | null = null;
  try {
    raw = new TextDecoder(encoding, { fatal: true }).decode(bytes, { stream: truncated });
  } catch {
    raw = new TextDecoder(encoding).decode(bytes, { stream: truncated });
    readOnlyReason = encoding === "utf-8" ? "notUtf8" : "utf16";
  }
  const crlf = raw.match(/\r\n/g)?.length ?? 0;
  const lf = (raw.match(/\n/g)?.length ?? 0) - crlf;
  return {
    text: raw.replace(/\r\n/g, "\n"),
    format: { encoding, bom, eol: crlf > lf ? "\r\n" : "\n" },
    readOnlyReason,
  };
}

export function encodeNote(text: string, format: NoteFormat): string {
  const lines = text.replace(/\r\n/g, "\n");
  const body = format.eol === "\r\n" ? lines.replace(/\n/g, "\r\n") : lines;
  return format.bom ? `\uFEFF${body}` : body;
}
