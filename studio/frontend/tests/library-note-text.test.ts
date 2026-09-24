// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Library notes open in an editor that saves back to the file. A file written by Notepad or
// PowerShell (UTF-16 with a BOM, CRLF endings) or in a legacy code page must never be rewritten
// into something else by opening and saving it.

import assert from "node:assert/strict";
import test from "node:test";

import { decodeNote, encodeNote } from "../src/features/library/note-text.ts";

const utf8 = (text: string) => new TextEncoder().encode(text);

function utf16le(text: string): Uint8Array {
  const bytes = new Uint8Array(2 + text.length * 2);
  bytes.set([0xff, 0xfe]);
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    bytes[2 + i * 2] = code & 0xff;
    bytes[3 + i * 2] = code >> 8;
  }
  return bytes;
}

test("plain UTF-8 is editable and saves back unchanged", () => {
  const note = decodeNote(utf8("héllo\nworld\n"));
  assert.equal(note.text, "héllo\nworld\n");
  assert.equal(note.readOnlyReason, null);
  assert.deepEqual(note.format, { encoding: "utf-8", bom: false, eol: "\n" });
  assert.equal(encodeNote(note.text, note.format), "héllo\nworld\n");
});

test("CRLF files show LF and save back as CRLF", () => {
  const note = decodeNote(utf8("a\r\nb\r\nc"));
  assert.equal(note.text, "a\nb\nc");
  assert.equal(note.format.eol, "\r\n");
  assert.equal(encodeNote("a\nb\nc\nd", note.format), "a\r\nb\r\nc\r\nd");
  // Already-CRLF text (a paste) is not doubled.
  assert.equal(encodeNote("x\r\ny", note.format), "x\r\ny");
});

test("a UTF-8 BOM is kept on save", () => {
  const note = decodeNote(new Uint8Array([0xef, 0xbb, 0xbf, ...utf8("hi")]));
  assert.equal(note.text, "hi");
  assert.equal(note.readOnlyReason, null);
  assert.equal(note.format.bom, true);
  assert.equal(encodeNote("hi!", note.format), "\uFEFFhi!");
});

test("UTF-16LE with a BOM decodes, and opens read-only", () => {
  const note = decodeNote(utf16le("Write-Host 'hi'\r\n"));
  assert.equal(note.text, "Write-Host 'hi'\n");
  assert.equal(note.format.encoding, "utf-16le");
  assert.equal(note.readOnlyReason, "utf16");
});

test("UTF-16BE with a BOM decodes", () => {
  const note = decodeNote(new Uint8Array([0xfe, 0xff, 0x00, 0x41, 0x00, 0x42]));
  assert.equal(note.text, "AB");
  assert.equal(note.format.encoding, "utf-16be");
});

test("a legacy code page is shown read-only rather than saved as mojibake", () => {
  // "café" in windows-1252.
  const note = decodeNote(new Uint8Array([0x63, 0x61, 0x66, 0xe9]));
  assert.equal(note.readOnlyReason, "notUtf8");
  assert.ok(note.text.startsWith("caf"));
});

test("a prefix cut inside a character is not read as a bad encoding", () => {
  const bytes = utf8("ab€");
  const note = decodeNote(bytes.subarray(0, bytes.length - 1), true);
  assert.equal(note.text, "ab");
  assert.equal(note.readOnlyReason, null);
});
