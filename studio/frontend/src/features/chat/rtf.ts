// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  MAX_OPEN_DOCUMENT_ARCHIVE_BYTES,
  MAX_OPEN_DOCUMENT_XML_BYTES,
} from "./open-document";

// A call, not a const: read at module scope, an import the chat barrel also reaches can still be
// in its temporal dead zone when this module runs.
function maxTextLength(): number {
  return MAX_OPEN_DOCUMENT_XML_BYTES;
}

const MAX_GROUP_DEPTH = 1024;
const LATIN1 = new TextDecoder("latin1");

// Groups whose text is not document body: tables, metadata, pictures, field codes, headers and footers.
const SKIPPED_DESTINATIONS = new Set([
  "fonttbl",
  "colortbl",
  "stylesheet",
  "listtable",
  "listoverridetable",
  "revtbl",
  "rsidtbl",
  "filetbl",
  "info",
  "pict",
  "nonshppict",
  "object",
  "objdata",
  "fldinst",
  "sn",
  "sv",
  "pn",
  "header",
  "headerl",
  "headerr",
  "headerf",
  "footer",
  "footerl",
  "footerr",
  "footerf",
  "footnote",
  "annotation",
  "xe",
  "tc",
  "bkmkstart",
  "bkmkend",
]);

const CONTROL_WORD_TEXT = new Map<string, string>([
  ["par", "\n"],
  ["line", "\n"],
  ["sect", "\n"],
  ["page", "\n"],
  ["row", "\n"],
  ["cell", "\t"],
  ["nestcell", "\t"],
  ["tab", "\t"],
  ["emdash", "—"],
  ["endash", "–"],
  ["bullet", "•"],
  ["lquote", "‘"],
  ["rquote", "’"],
  ["ldblquote", "“"],
  ["rdblquote", "”"],
  ["emspace", " "],
  ["enspace", " "],
  ["qmspace", " "],
]);

// \fcharset values mapped to Windows code pages.
const CHARSET_CODEPAGES: Record<number, number> = {
  0: 1252,
  77: 10000,
  128: 932,
  129: 949,
  134: 936,
  136: 950,
  161: 1253,
  162: 1254,
  163: 1258,
  177: 1255,
  178: 1256,
  186: 1257,
  204: 1251,
  222: 874,
  238: 1250,
};

const CODEPAGE_ENCODINGS: Record<number, string> = {
  874: "windows-874",
  932: "shift_jis",
  936: "gbk",
  949: "euc-kr",
  950: "big5",
  10000: "macintosh",
  65001: "utf-8",
};

type Group = {
  skip: boolean;
  fontTable: boolean;
  inTable: boolean;
  uc: number;
  font: number;
};

export async function readRtfAttachmentContent(
  file: File,
  filename: string,
): Promise<{ label: "RTF"; text: string }> {
  if (file.size > MAX_OPEN_DOCUMENT_ARCHIVE_BYTES) {
    throw new Error(`RTF file is too large: ${filename}`);
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (LATIN1.decode(bytes.subarray(0, 5)) !== "{\\rtf") {
    throw new Error(`Not an RTF file: ${filename}`);
  }
  return { label: "RTF", text: rtfToText(bytes, filename) };
}

function encodingFor(codepage: number): string {
  if (codepage >= 1250 && codepage <= 1258) {
    return `windows-${codepage}`;
  }
  return CODEPAGE_ENCODINGS[codepage] ?? "windows-1252";
}

function rtfToText(bytes: Uint8Array, filename: string): string {
  const maxLength = maxTextLength();
  const fonts = new Map<number, number>();
  let defaultCodepage = 1252;
  let defaultFont = -1;
  let group: Group = {
    skip: false,
    fontTable: false,
    inTable: false,
    uc: 1,
    font: -1,
  };
  const stack: Group[] = [];
  const parts: string[] = [];
  let length = 0;
  let pending: number[] = [];
  let pendingEncoding = "";
  let fallbackToSkip = 0;
  let tableFont = -1;

  const flush = () => {
    if (pending.length > 0) {
      // Charged when buffered: a byte decodes to at most one character.
      parts.push(
        new TextDecoder(pendingEncoding).decode(new Uint8Array(pending)),
      );
      pending = [];
    }
  };
  const append = (text: string) => {
    parts.push(text);
    length += text.length;
  };
  const emit = (text: string) => {
    if (!group.skip) {
      flush();
      append(text);
    }
  };
  const emitByte = (byte: number) => {
    if (group.skip) {
      return;
    }
    const charset = fonts.get(group.font < 0 ? defaultFont : group.font);
    const encoding = encodingFor(
      charset === undefined || charset === 1
        ? defaultCodepage
        : (CHARSET_CODEPAGES[charset] ?? defaultCodepage),
    );
    if (encoding !== pendingEncoding) {
      flush();
      pendingEncoding = encoding;
    }
    pending.push(byte);
    length++;
  };
  // Writers escape a double-byte lead byte but may leave its trail byte literal, even \\ { or }.
  const emitLiteral = (byte: number) => {
    if (pending.length > 0) {
      emitByte(byte);
      flush();
    } else {
      emit(String.fromCharCode(byte));
    }
  };

  let index = 0;
  while (index < bytes.length && length < maxLength) {
    const byte = bytes[index++];
    if (byte === 0x7b) {
      if (stack.length >= MAX_GROUP_DEPTH) {
        throw new Error(`RTF groups nest too deeply: ${filename}`);
      }
      stack.push(group);
      group = { ...group };
      fallbackToSkip = 0;
    } else if (byte === 0x7d) {
      group = stack.pop() ?? group;
      fallbackToSkip = 0;
    } else if (byte === 0x0d || byte === 0x0a) {
      continue;
    } else if (byte !== 0x5c) {
      if (fallbackToSkip > 0) {
        fallbackToSkip--;
      } else if (byte >= 0x80) {
        emitByte(byte);
      } else if (pending.length > 0) {
        emitLiteral(byte);
      } else {
        const start = index - 1;
        while (index < bytes.length && isPlainText(bytes[index])) index++;
        emit(LATIN1.decode(bytes.subarray(start, index)));
      }
    } else if (index < bytes.length && isLetter(bytes[index])) {
      const start = index;
      while (index < bytes.length && isLetter(bytes[index])) index++;
      const word = LATIN1.decode(bytes.subarray(start, index));
      let param: number | null = null;
      const paramStart = index;
      if (bytes[index] === 0x2d) index++;
      while (index < bytes.length && isDigit(bytes[index])) index++;
      if (index > paramStart && bytes[index - 1] !== 0x2d) {
        param = Number(LATIN1.decode(bytes.subarray(paramStart, index)));
      } else {
        index = paramStart;
      }
      if (bytes[index] === 0x20) index++;

      if (word === "bin" && param !== null) {
        index += Math.max(0, param);
      } else if (fallbackToSkip > 0) {
        fallbackToSkip--;
      } else if (word === "u" && param !== null) {
        emit(String.fromCharCode(param < 0 ? param + 0x10000 : param));
        fallbackToSkip = group.uc;
      } else if (word === "uc" && param !== null) {
        group.uc = Math.max(0, param);
      } else if (word === "deff" && param !== null) {
        defaultFont = param;
      } else if (word === "ansicpg" && param !== null) {
        defaultCodepage = param;
      } else if (word === "f" && param !== null) {
        if (group.fontTable) {
          tableFont = param;
        } else {
          group.font = param;
        }
      } else if (word === "fcharset" && param !== null) {
        fonts.set(tableFont, param);
      } else if (word === "plain") {
        group.font = -1;
      } else if (word === "intbl" || word === "pard") {
        group.inTable = word === "intbl";
      } else if (word === "par" && group.inTable) {
        // A paragraph break inside a cell would split the row across lines.
        emit(" ");
      } else if (SKIPPED_DESTINATIONS.has(word)) {
        group.skip = true;
        group.fontTable ||= word === "fonttbl";
      } else if (CONTROL_WORD_TEXT.has(word)) {
        emit(CONTROL_WORD_TEXT.get(word) as string);
      }
    } else if (index < bytes.length) {
      const symbol = bytes[index++];
      if (symbol === 0x27) {
        const byteValue = parseInt(
          String.fromCharCode(bytes[index], bytes[index + 1]),
          16,
        );
        index += 2;
        if (fallbackToSkip > 0) {
          fallbackToSkip--;
        } else if (!Number.isNaN(byteValue)) {
          emitByte(byteValue);
        }
      } else if (symbol === 0x2a) {
        group.skip = true;
      } else if (fallbackToSkip > 0) {
        fallbackToSkip--;
      } else if (symbol === 0x0d || symbol === 0x0a) {
        emit("\n");
      } else if (symbol === 0x7e) {
        emit(" ");
      } else if (symbol === 0x5f) {
        emit("-");
      } else if (symbol === 0x5c || symbol === 0x7b || symbol === 0x7d) {
        emitLiteral(symbol);
      }
    }
  }
  flush();

  const text = parts
    .join("")
    .slice(0, maxLength)
    .replace(/[^\S\n]+\n/g, "\n")
    .replace(/ +\t/g, "\t")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return length >= maxLength
    ? `${text}\n\n[Truncated: the document has more text than one attachment carries]`
    : text;
}

function isLetter(byte: number): boolean {
  return (byte >= 0x61 && byte <= 0x7a) || (byte >= 0x41 && byte <= 0x5a);
}

function isPlainText(byte: number): boolean {
  return (
    byte < 0x80 &&
    byte !== 0x5c &&
    byte !== 0x7b &&
    byte !== 0x7d &&
    byte !== 0x0d &&
    byte !== 0x0a
  );
}

function isDigit(byte: number): boolean {
  return byte >= 0x30 && byte <= 0x39;
}
