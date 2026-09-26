// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Unzipped, strFromU8, unzipSync } from "fflate";

/** Readers for the parts of an XLSX or PPTX a viewer shows: values, not a faithful rendering. */

const MAX_UNPACKED_BYTES = 200 * 1024 * 1024;
export const MAX_SHEET_ROWS = 5000;
export const MAX_SHEET_COLUMNS = 200;

export interface SheetCell {
  text: string;
  numeric?: boolean;
  bold?: boolean;
  italic?: boolean;
}

export interface Sheet {
  name: string;
  rows: (SheetCell | undefined)[][];
  /** Column widths in pixels, where the file sets them. */
  widths: (number | undefined)[];
  truncated: boolean;
}

function unpack(bytes: Uint8Array, wanted: (name: string) => boolean): Unzipped {
  let total = 0;
  return unzipSync(bytes, {
    filter: (entry) => {
      if (!wanted(entry.name)) return false;
      total += entry.originalSize;
      if (total > MAX_UNPACKED_BYTES) throw new Error("File is too large to preview.");
      return true;
    },
  });
}

function xml(files: Unzipped, path: string): Document | null {
  const bytes = files[path];
  if (!bytes) return null;
  const doc = new DOMParser().parseFromString(strFromU8(bytes), "application/xml");
  return doc.getElementsByTagName("parsererror").length ? null : doc;
}

/** Elements by local name, whatever prefix the writer gave the namespace. */
function all(node: Document | Element, name: string): Element[] {
  return Array.from(node.getElementsByTagNameNS("*", name));
}

function first(node: Document | Element, name: string): Element | undefined {
  return node.getElementsByTagNameNS("*", name)[0];
}

function children(node: Element, name: string): Element[] {
  return Array.from(node.children).filter((child) => child.localName === name);
}

/** Relationship id to part path, resolved against the part the .rels file belongs to. */
function relationships(files: Unzipped, part: string): Map<string, string> {
  const slash = part.lastIndexOf("/");
  const dir = part.slice(0, slash + 1);
  const doc = xml(files, `${dir}_rels/${part.slice(slash + 1)}.rels`);
  const out = new Map<string, string>();
  for (const rel of doc ? all(doc, "Relationship") : []) {
    const target = rel.getAttribute("Target") ?? "";
    const id = rel.getAttribute("Id");
    if (id && rel.getAttribute("TargetMode") !== "External") out.set(id, resolvePath(dir, target));
  }
  return out;
}

function resolvePath(dir: string, target: string): string {
  const parts = (target.startsWith("/") ? target.slice(1) : dir + target).split("/");
  const out: string[] = [];
  for (const part of parts) {
    if (part === "..") out.pop();
    else if (part !== "." && part !== "") out.push(part);
  }
  return out.join("/");
}

/** An `r:`-prefixed attribute, by its namespace: a slide id carries both `id` and `r:id`. */
function relId(node: Element, name: string): string | null {
  for (const attr of Array.from(node.attributes)) {
    if (attr.localName === name && attr.namespaceURI?.endsWith("/relationships")) return attr.value;
  }
  return null;
}

// ---------------------------------------------------------------------------------------------
// Spreadsheets

const BUILTIN_FORMATS: Record<number, string> = {
  1: "0",
  2: "0.00",
  3: "#,##0",
  4: "#,##0.00",
  9: "0%",
  10: "0.00%",
  11: "0.00E+00",
  14: "m/d/yyyy",
  15: "d-mmm-yy",
  16: "d-mmm",
  17: "mmm-yy",
  18: "h:mm AM/PM",
  19: "h:mm:ss AM/PM",
  20: "h:mm",
  21: "h:mm:ss",
  22: "m/d/yyyy h:mm",
  37: "#,##0 ;(#,##0)",
  38: "#,##0 ;[Red](#,##0)",
  39: "#,##0.00;(#,##0.00)",
  40: "#,##0.00;[Red](#,##0.00)",
  49: "@",
};

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function formatDate(serial: number, code: string): string {
  const date = new Date(Math.round((serial - 25569) * 86400000));
  const y = date.getUTCFullYear();
  const m = date.getUTCMonth();
  const d = date.getUTCDate();
  const pad = (n: number) => String(n).padStart(2, "0");
  let time = "";
  if (/h/.test(code)) {
    const hours = date.getUTCHours();
    const twelve = /am\/pm/.test(code);
    time = `${twelve ? hours % 12 || 12 : hours}:${pad(date.getUTCMinutes())}`;
    if (/:ss/.test(code)) time += `:${pad(date.getUTCSeconds())}`;
    if (twelve) time += hours < 12 ? " AM" : " PM";
  }
  // A time of day alone (h:mm) has no date part to show.
  if (time && !/[dy]/.test(code)) return time;
  let out = `${m + 1}/${d}/${y}`;
  if (/mmm/.test(code)) out = `${d}-${MONTHS[m]}-${String(y).slice(-2)}`;
  else if (/^y/.test(code)) out = `${y}-${pad(m + 1)}-${pad(d)}`;
  else if (/^d/.test(code)) out = `${d}/${m + 1}/${y}`;
  return time ? `${out} ${time}` : out;
}

/** The common shapes of an Excel number format: decimals, grouping, percent, currency, dates. */
export function formatNumber(value: number, rawCode: string | undefined): string {
  if (!rawCode || rawCode === "General" || rawCode === "@") {
    return String(Number.isInteger(value) ? value : Number(value.toPrecision(11)));
  }
  const section = rawCode.split(";")[value < 0 && rawCode.includes(";") ? 1 : 0] ?? rawCode;
  // Quoted text, escapes and [$€-407]-style currency tags keep their symbol; [Red] and the like go.
  const code = section
    .replace(/\[\$([^\]-]*)[^\]]*\]/g, "$1")
    .replace(/\[[^\]]*\]/g, "")
    .replace(/"([^"]*)"/g, "$1")
    .replace(/\\(.)/g, "$1")
    .replace(/_.|\*./g, "");
  if (!/[0#?]/.test(code) && /[dmyhs]/i.test(code)) {
    return formatDate(value, code.toLowerCase());
  }
  if (/E\+/i.test(code)) {
    // Excel's form: 1.23E+03, a two-digit exponent and an upper-case E.
    const digits = (code.split(/E/i)[0]!.split(".")[1]?.match(/0/g) ?? []).length;
    return value.toExponential(digits).toUpperCase().replace(/E([+-])(\d)$/, "E$10$2");
  }
  const percent = code.includes("%");
  const number = Math.abs(percent ? value * 100 : value);
  const decimals = (code.split(".")[1]?.match(/[0#]/g) ?? []).length;
  const digits = number.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
    useGrouping: code.includes(","),
  });
  const firstDigit = code.search(/[0#?]/);
  const lastDigit = Math.max(code.lastIndexOf("0"), code.lastIndexOf("#"), code.lastIndexOf("?"));
  const prefix = firstDigit > 0 ? code.slice(0, firstDigit).replace(/[,.]/g, "") : "";
  const suffix = lastDigit >= 0 ? code.slice(lastDigit + 1).replace(/[,.]/g, "") : "";
  const parenthesised = /\(.*\)/.test(section);
  const sign = value < 0 && !rawCode.includes(";") ? "-" : "";
  const body = `${prefix}${digits}${suffix}`.trim();
  return parenthesised && value < 0 ? body : `${sign}${body}`;
}

function columnIndex(ref: string): number {
  let index = 0;
  for (const char of ref) {
    const code = char.charCodeAt(0);
    if (code < 65 || code > 90) break;
    index = index * 26 + (code - 64);
  }
  return index - 1;
}

export function columnName(index: number): string {
  let name = "";
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) {
    name = String.fromCharCode(65 + ((n - 1) % 26)) + name;
  }
  return name;
}

function inlineText(node: Element): string {
  // Phonetic runs (<rPh>) are ruby annotations, not part of the value.
  return all(node, "t")
    .filter((t) => t.parentElement?.localName !== "rPh")
    .map((t) => t.textContent ?? "")
    .join("");
}

interface CellStyle {
  format?: string;
  bold?: boolean;
  italic?: boolean;
}

function readStyles(doc: Document | null): CellStyle[] {
  if (!doc) return [];
  const formats = new Map<number, string>();
  for (const fmt of all(doc, "numFmt")) {
    formats.set(Number(fmt.getAttribute("numFmtId")), fmt.getAttribute("formatCode") ?? "");
  }
  const fontsNode = first(doc, "fonts");
  const fonts = fontsNode
    ? children(fontsNode, "font").map((font) => ({
        bold: children(font, "b").some((b) => b.getAttribute("val") !== "0"),
        italic: children(font, "i").some((i) => i.getAttribute("val") !== "0"),
      }))
    : [];
  const xfs = first(doc, "cellXfs");
  return (xfs ? children(xfs, "xf") : []).map((xf) => {
    const id = Number(xf.getAttribute("numFmtId") ?? 0);
    const font = fonts[Number(xf.getAttribute("fontId") ?? 0)];
    return { format: formats.get(id) ?? BUILTIN_FORMATS[id], ...font };
  });
}

function readSheet(doc: Document, name: string, strings: string[], styles: CellStyle[]): Sheet {
  const rows: (SheetCell | undefined)[][] = [];
  let truncated = false;
  let nextRow = 0;
  for (const row of all(doc, "row")) {
    const r = Number(row.getAttribute("r") ?? nextRow + 1) - 1;
    nextRow = r + 1;
    if (r >= MAX_SHEET_ROWS) {
      truncated = true;
      break;
    }
    const cells: (SheetCell | undefined)[] = [];
    let nextColumn = 0;
    for (const c of children(row, "c")) {
      const ref = c.getAttribute("r");
      const col = ref ? columnIndex(ref) : nextColumn;
      nextColumn = col + 1;
      if (col >= MAX_SHEET_COLUMNS) {
        truncated = true;
        continue;
      }
      const type = c.getAttribute("t");
      const raw = first(c, "v")?.textContent ?? "";
      const style = styles[Number(c.getAttribute("s") ?? 0)] ?? {};
      let text = raw;
      let numeric = false;
      if (type === "s") text = strings[Number(raw)] ?? "";
      else if (type === "inlineStr") text = inlineText(c);
      else if (type === "b") text = raw === "1" ? "TRUE" : "FALSE";
      else if (type !== "str" && type !== "e" && raw !== "" && Number.isFinite(Number(raw))) {
        text = formatNumber(Number(raw), style.format);
        numeric = true;
      }
      if (text === "" && !style.bold) continue;
      cells[col] = { text, numeric, bold: style.bold, italic: style.italic };
    }
    rows[r] = cells;
  }
  const widths: (number | undefined)[] = [];
  for (const col of all(doc, "col")) {
    const width = Number(col.getAttribute("width"));
    if (!width || col.getAttribute("hidden") === "1") continue;
    const min = Number(col.getAttribute("min") ?? 1);
    const max = Math.min(Number(col.getAttribute("max") ?? min), MAX_SHEET_COLUMNS);
    for (let i = min; i <= max; i++) widths[i - 1] = Math.round(width * 7 + 5);
  }
  return { name, rows, widths, truncated };
}

export function readXlsx(bytes: Uint8Array): Sheet[] {
  const files = unpack(bytes, (name) => name.endsWith(".xml") || name.endsWith(".rels"));
  const workbook = xml(files, "xl/workbook.xml");
  if (!workbook) throw new Error("Not a valid XLSX workbook.");
  const rels = relationships(files, "xl/workbook.xml");
  const sharedDoc = xml(files, "xl/sharedStrings.xml");
  const strings = sharedDoc ? all(sharedDoc, "si").map(inlineText) : [];
  const styles = readStyles(xml(files, "xl/styles.xml"));
  const sheets: Sheet[] = [];
  for (const sheet of all(workbook, "sheet")) {
    if (sheet.getAttribute("state") === "hidden" || sheet.getAttribute("state") === "veryHidden") continue;
    const path = rels.get(relId(sheet, "id") ?? "");
    const doc = path ? xml(files, path) : null;
    if (doc) sheets.push(readSheet(doc, sheet.getAttribute("name") ?? "Sheet", strings, styles));
  }
  return sheets;
}

/** RFC 4180 fields: quoted values may hold the delimiter, quotes ("") and line breaks. */
export function readDelimited(text: string, delimiter: string, name: string): Sheet {
  const rows: (SheetCell | undefined)[][] = [];
  let row: (SheetCell | undefined)[] = [];
  let field = "";
  let quoted = false;
  let truncated = false;
  const push = () => {
    if (row.length < MAX_SHEET_COLUMNS) {
      const numeric = field.trim() !== "" && Number.isFinite(Number(field.replace(/[$,%]/g, "")));
      row.push(field === "" ? undefined : { text: field, numeric });
    } else truncated = true;
    field = "";
  };
  for (let i = 0; i < text.length; i++) {
    const char = text[i]!;
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"' && field === "") quoted = true;
    else if (char === delimiter) push();
    else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[i + 1] === "\n") i++;
      push();
      rows.push(row);
      row = [];
      if (rows.length >= MAX_SHEET_ROWS) {
        truncated = i < text.length - 1;
        break;
      }
    } else field += char;
  }
  if (field !== "" || row.length) {
    push();
    rows.push(row);
  }
  return { name, rows, widths: [], truncated };
}

// ---------------------------------------------------------------------------------------------
// Presentations

export interface SlideBox {
  /** Position and size as fractions of the slide, when the shape sets its own. */
  frame?: { x: number; y: number; w: number; h: number };
  placeholder?: string;
  paragraphs?: { text: string; size?: number; bold?: boolean; align?: string; bullet?: boolean }[];
  image?: string;
}

export interface Slide {
  boxes: SlideBox[];
}

export interface Deck {
  /** Height over width. */
  aspect: number;
  /** Slide width in points, so a run's size can be scaled with the slide. */
  widthPt: number;
  slides: Slide[];
}

const IMAGE_TYPES: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  svg: "image/svg+xml",
  webp: "image/webp",
  bmp: "image/bmp",
};

/** A picture as a data URL: nothing to revoke, however often the deck is mounted and unmounted. */
function dataUrl(bytes: Uint8Array, type: string): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return `data:${type};base64,${btoa(binary)}`;
}

function readFrame(shape: Element, cx: number, cy: number): SlideBox["frame"] {
  const xfrm = first(shape, "xfrm");
  const off = xfrm && first(xfrm, "off");
  const ext = xfrm && first(xfrm, "ext");
  if (!off || !ext) return undefined;
  return {
    x: Number(off.getAttribute("x")) / cx,
    y: Number(off.getAttribute("y")) / cy,
    w: Number(ext.getAttribute("cx")) / cx,
    h: Number(ext.getAttribute("cy")) / cy,
  };
}

export function readPptx(bytes: Uint8Array): Deck {
  const files = unpack(bytes, (name) => name.startsWith("ppt/") || name.startsWith("_rels/"));
  const presentation = xml(files, "ppt/presentation.xml");
  if (!presentation) throw new Error("Not a valid PPTX presentation.");
  const size = first(presentation, "sldSz");
  const cx = Number(size?.getAttribute("cx")) || 12192000;
  const cy = Number(size?.getAttribute("cy")) || 6858000;
  const rels = relationships(files, "ppt/presentation.xml");
  const slides: Slide[] = [];
  for (const id of all(presentation, "sldId")) {
    const path = rels.get(relId(id, "id") ?? "");
    const doc = path ? xml(files, path) : null;
    if (!path || !doc) continue;
    const slideRels = relationships(files, path);
    const boxes: SlideBox[] = [];
    for (const shape of all(doc, "sp")) {
      const body = first(shape, "txBody");
      if (!body) continue;
      const paragraphs = all(body, "p")
        .map((p) => {
          const runs = all(p, "r");
          const rPr = runs[0] && first(runs[0], "rPr");
          const pPr = first(p, "pPr");
          const sz = Number(rPr?.getAttribute("sz"));
          return {
            text: all(p, "t").map((t) => t.textContent ?? "").join(""),
            size: sz ? sz / 100 : undefined,
            bold: rPr?.getAttribute("b") === "1",
            align: pPr?.getAttribute("algn") ?? undefined,
            bullet: Boolean(pPr && (first(pPr, "buChar") || first(pPr, "buAutoNum"))),
          };
        })
        .filter((p) => p.text.trim());
      if (!paragraphs.length) continue;
      boxes.push({
        frame: readFrame(shape, cx, cy),
        placeholder: first(shape, "ph")?.getAttribute("type") ?? (first(shape, "ph") ? "body" : undefined),
        paragraphs,
      });
    }
    for (const pic of all(doc, "pic")) {
      const blip = first(pic, "blip");
      const target = blip && slideRels.get(relId(blip, "embed") ?? "");
      const data = target && files[target];
      const type = target && IMAGE_TYPES[target.split(".").pop()!.toLowerCase()];
      if (!data || !type) continue;
      boxes.unshift({ frame: readFrame(pic, cx, cy), image: dataUrl(data, type) });
    }
    slides.push({ boxes });
  }
  return { aspect: cy / cx, widthPt: cx / 12700, slides };
}
