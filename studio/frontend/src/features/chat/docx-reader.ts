// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { codePoint } from "dingbat-to-unicode";

export { default as mammoth } from "mammoth";

const DOCX_SYMBOL_FONTS = new Set(["symbol", "wingdings", "wingdings 2", "wingdings 3", "webdings"]);
const DOCX_SYMBOL_HEX_RE = /^[0-9a-f]+$/i;
const UNPRINTABLE_RE = /[\p{C}\p{Z}]/u;

/** Insert > Symbol character; mirrors the backend's _docx_symbol. */
export function docxSymbolText(element: Element, w: string): string {
  const raw = element.getAttributeNS(w, "char") ?? "";
  if (!DOCX_SYMBOL_HEX_RE.test(raw)) return "";
  let code = Number.parseInt(raw, 16);
  const font = element.getAttributeNS(w, "font") ?? "";
  const page = Math.floor(code / 0x100);
  if (DOCX_SYMBOL_FONTS.has(font.toLowerCase()) && (page === 0 || page === 0xf0)) {
    code = codePoint(font, code % 0x100)?.codePoint ?? 0;
  }
  if (code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return "";
  const char = String.fromCodePoint(code);
  return char === " " || !UNPRINTABLE_RE.test(char) ? char : "";
}
