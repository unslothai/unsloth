// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { codePoint } from "dingbat-to-unicode";
import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";

export { default as mammoth } from "mammoth";

const WORDPROCESSINGML_NAMESPACES = new Set([
  "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
  "http://purl.oclc.org/ooxml/wordprocessingml/main",
]);
const DOCX_SYMBOL_FONTS = new Set(["symbol", "wingdings", "wingdings 2", "wingdings 3", "webdings"]);
const DOCX_SYMBOL_HEX_RE = /^[0-9a-f]+$/i;
const UNPRINTABLE_RE = /[\p{C}\p{Z}]/u;

/** Insert > Symbol character; mirrors the backend's _docx_symbol. */
function docxSymbolText(element: Element, w: string): string {
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

/** Each w:sym becomes the text it shows, so the body, equation and note readers all see it.
 *  Parts are picked by size, not name: mammoth reaches them through relationships. */
export function writeDocxSymbols(archive: Uint8Array, maxPartBytes: number): Uint8Array {
  const rewritten: Record<string, Uint8Array> = {};
  const parts = unzipSync(archive, { filter: (entry) => entry.originalSize <= maxPartBytes });
  for (const [name, bytes] of Object.entries(parts)) {
    const xml = strFromU8(bytes);
    if (!xml.includes("sym")) continue;
    const doc = new DOMParser().parseFromString(xml, "application/xml");
    const w = doc.documentElement?.namespaceURI ?? "";
    if (!WORDPROCESSINGML_NAMESPACES.has(w) || doc.getElementsByTagName("parsererror").length) continue;
    const symbols = Array.from(doc.getElementsByTagNameNS(w, "sym"));
    for (const symbol of symbols) {
      const run = symbol.parentNode as Element;
      const text = doc.createElementNS(run.namespaceURI, run.prefix ? `${run.prefix}:t` : "t");
      text.setAttributeNS("http://www.w3.org/XML/1998/namespace", "xml:space", "preserve");
      text.appendChild(doc.createTextNode(docxSymbolText(symbol, w)));
      run.replaceChild(text, symbol);
    }
    if (symbols.length) rewritten[name] = strToU8(new XMLSerializer().serializeToString(doc));
  }
  if (!Object.keys(rewritten).length) return archive;
  return zipSync({ ...unzipSync(archive), ...rewritten }, { level: 0 });
}
