// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Unzipped, strFromU8, unzipSync } from "fflate";

/** Readers for the parts of an XLSX or PPTX a viewer shows: values, not a faithful rendering. */

const MAX_UNPACKED_BYTES = 200 * 1024 * 1024;
export const MAX_SHEET_ROWS = 5000;
export const MAX_SHEET_COLUMNS = 200;
// Across every sheet, so a workbook of many sheets reads no more than one full one.
const MAX_WORKBOOK_CELLS = MAX_SHEET_ROWS * MAX_SHEET_COLUMNS;

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
  /** Rows and columns the workbook hides, by index. Their cells are not read. */
  hidden?: { rows: Set<number>; columns: Set<number> };
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
  45: "mm:ss",
  46: "[h]:mm:ss",
  47: "mm:ss.0",
  49: "@",
};

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const DATE_TOKEN = /"[^"]*"|\\.|y+|m+|d+|h+|s+|am\/pm|a\/p|\.0+|./gi;

/** A date or time, token by token as the format writes it. `m` is minutes after an hour or before
 *  a second, otherwise the month. */
function formatDate(serial: number, code: string): string {
  const fraction = /s\.(0+)/i.exec(code)?.[1]?.length ?? 0;
  const unit = 1000 / 10 ** fraction;
  // The 1900 system counts a 29 February 1900 that never was (serial 60), so earlier serials run a day ahead.
  const whole = Math.floor(serial);
  const shifted = serial < 60 ? serial + 1 : serial;
  const date = new Date(Math.round(((shifted - 25569) * 86400000) / unit) * unit);
  const tokens = code.match(DATE_TOKEN) ?? [];
  const kinds = tokens.map((token) => /^[ymdhs]/i.test(token) && !token.startsWith("\\") ? token[0]!.toLowerCase() : "");
  const twelve = tokens.some((token) => /^(am\/pm|a\/p)$/i.test(token));
  const pad = (n: number, width = 2) => String(n).padStart(width, "0");
  const hours = date.getUTCHours();
  return tokens
    .map((token, index) => {
      const n = token.length;
      switch (kinds[index]) {
        case "y":
          return n <= 2 ? pad(date.getUTCFullYear() % 100) : String(date.getUTCFullYear());
        case "d": {
          // Weekdays by serial, as Excel counts them: serial 1 is a Sunday.
          const weekday = DAYS[(((whole + 6) % 7) + 7) % 7]!;
          const day = whole === 60 ? 29 : date.getUTCDate();
          if (n >= 4) return weekday;
          if (n === 3) return weekday.slice(0, 3);
          return n === 2 ? pad(day) : String(day);
        }
        case "h": {
          const h = twelve ? hours % 12 || 12 : hours;
          return n >= 2 ? pad(h) : String(h);
        }
        case "s":
          return n >= 2 ? pad(date.getUTCSeconds()) : String(date.getUTCSeconds());
        case "m": {
          const previous = kinds.slice(0, index).reverse().find(Boolean);
          const next = kinds.slice(index + 1).find(Boolean);
          if (n <= 2 && (previous === "h" || next === "s")) {
            return n === 2 ? pad(date.getUTCMinutes()) : String(date.getUTCMinutes());
          }
          const month = date.getUTCMonth();
          if (n >= 5) return MONTHS[month]![0]!;
          if (n === 4) return MONTHS[month]!;
          if (n === 3) return MONTHS[month]!.slice(0, 3);
          return n === 2 ? pad(month + 1) : String(month + 1);
        }
      }
      if (/^am\/pm$/i.test(token)) return hours < 12 ? "AM" : "PM";
      if (/^a\/p$/i.test(token)) return hours < 12 ? "A" : "P";
      if (/^\.0+$/.test(token)) return `.${pad(date.getUTCMilliseconds(), 3).slice(0, n - 1)}`;
      if (token.startsWith('"')) return token.slice(1, -1);
      return token.startsWith("\\") ? token.slice(1) : token;
    })
    .join("");
}

/** A fraction format: # ?/? (whole part and fraction), ?/? (improper), or a fixed denominator (# ?/8).
 *  The denominator is the closest one its placeholders allow. Null when the format has none. */
function formatFraction(value: number, code: string): string | null {
  const match = /(?:([0#?]+)\s+)?[0#?]+\s*\/\s*([1-9]\d*|[0#?]+)/.exec(code);
  if (!match) return null;
  const text = match[0];
  const wholeCode = match[1];
  const denominatorCode = match[2] ?? "";
  const abs = Math.abs(value);
  let whole = wholeCode ? Math.floor(abs) : 0;
  const rest = abs - whole;
  let numerator = Math.round(rest);
  let denominator = 1;
  if (/^[1-9]/.test(denominatorCode)) {
    denominator = Number(denominatorCode);
    numerator = Math.round(rest * denominator);
  } else {
    let best = Math.abs(rest - numerator);
    for (let d = 2; d < 10 ** denominatorCode.length; d++) {
      const n = Math.round(rest * d);
      if (Math.abs(rest - n / d) < best - 1e-12) {
        best = Math.abs(rest - n / d);
        numerator = n;
        denominator = d;
      }
    }
  }
  if (wholeCode && numerator === denominator) {
    whole += 1;
    numerator = 0;
  }
  const fraction = numerator ? `${numerator}/${denominator}` : "";
  let body = `${numerator}/${denominator}`;
  if (wholeCode) body = whole || fraction ? [whole ? String(whole) : "", fraction].filter(Boolean).join(" ") : "0";
  const at = code.indexOf(text);
  return `${value < 0 ? "-" : ""}${code.slice(0, at)}${body}${code.slice(at + text.length)}`.trim();
}

/** An elapsed-time format ([h]:mm:ss, [mm]:ss, [ss]): the bracketed unit counts past its usual range. */
function formatElapsed(value: number, code: string): string {
  const total = Math.round(Math.abs(value) * 86400);
  const unit = /\[h+\]/i.test(code) ? 3600 : /\[m+\]/i.test(code) ? 60 : 1;
  const lead = Math.floor(total / unit);
  const pad = (n: number) => String(n).padStart(2, "0");
  const out = code
    .replace(/\[[hms]+\]/i, String(lead))
    .replace(/m+/i, pad(Math.floor((total % 3600) / 60)))
    .replace(/s+/i, pad(total % 60));
  return value < 0 ? `-${out}` : out;
}

/** The common shapes of an Excel number format: decimals, grouping, percent, currency, dates.
 *  `date1904`: the workbook counts dates from 1904, 1,462 days after the 1900 system. */
export function formatNumber(value: number, rawCode: string | undefined, date1904 = false): string {
  if (!rawCode || rawCode === "General" || rawCode === "@") {
    // 15 significant digits, as Excel stores: drops binary noise such as 0.30000000000000004.
    return String(Number.isInteger(value) ? value : Number(value.toPrecision(15)));
  }
  const section = rawCode.split(";")[value < 0 && rawCode.includes(";") ? 1 : 0] ?? rawCode;
  // [$€-407]-style currency tags keep their symbol; [Red] and the like go, but not the
  // elapsed-time units [h], [m] and [s]. Padding (_x) and fill (*x) go too.
  const tagged = section
    .replace(/\[\$([^\]-]*)[^\]]*\]/g, "$1")
    .replace(/\[(?![hms]+\])[^\]]*\]/gi, "")
    .replace(/_.|\*./g, "");
  // Quoted text and escapes are literal.
  const code = tagged.replace(/"([^"]*)"/g, "$1").replace(/\\(.)/g, "$1");
  // Fractional seconds (ss.0) are a time, not a number.
  const dateLike = !/[0#?]/.test(code.replace(/s\.0+/gi, "s"));
  if (dateLike && /\[[hms]+\]/i.test(code)) return formatElapsed(value, code.toLowerCase());
  if (dateLike && /[dmyhs]/i.test(code)) return formatDate(date1904 ? value + 1462 : value, tagged);
  if (/E\+/i.test(code)) {
    // Excel's form: 1.23E+03, a two-digit exponent and an upper-case E.
    const digits = (code.split(/E/i)[0]!.split(".")[1]?.match(/0/g) ?? []).length;
    return value.toExponential(digits).toUpperCase().replace(/E([+-])(\d)$/, "E$10$2");
  }
  const fraction = formatFraction(value, code);
  if (fraction !== null) return fraction;
  const percent = code.includes("%");
  // Commas after the last digit placeholder scale by a thousand each: #,##0,, shows millions.
  const scale = 1000 ** (code.match(/[0#?](,+)(?:\.|[^0#?]*$)/)?.[1]?.length ?? 0);
  const number = Math.abs(percent ? value * 100 : value) / scale;
  const decimals = (code.split(".")[1]?.match(/[0#]/g) ?? []).length;
  const digits = number.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
    useGrouping: /[0#?],[0#?]/.test(code),
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

function readSheet(
  doc: Document,
  name: string,
  strings: string[],
  styles: CellStyle[],
  date1904: boolean,
  budget: { cells: number },
): Sheet {
  const rows: (SheetCell | undefined)[][] = [];
  const widths: (number | undefined)[] = [];
  const hidden = { rows: new Set<number>(), columns: new Set<number>() };
  const isHidden = (node: Element) => ["1", "true"].includes(node.getAttribute("hidden") ?? "");
  for (const col of all(doc, "col")) {
    const min = Number(col.getAttribute("min") ?? 1);
    const max = Math.min(Number(col.getAttribute("max") ?? min), MAX_SHEET_COLUMNS);
    const width = Number(col.getAttribute("width"));
    for (let i = min; i <= max; i++) {
      if (isHidden(col)) hidden.columns.add(i - 1);
      else if (width) widths[i - 1] = Math.round(width * 7 + 5);
    }
  }
  let truncated = false;
  let nextRow = 0;
  for (const row of all(doc, "row")) {
    const r = Number(row.getAttribute("r") ?? nextRow + 1) - 1;
    nextRow = r + 1;
    if (r >= MAX_SHEET_ROWS || budget.cells <= 0) {
      truncated = true;
      break;
    }
    budget.cells--;
    // Hidden, as Excel shows it: neither in the grid nor in the text sent to the model.
    if (isHidden(row)) {
      hidden.rows.add(r);
      continue;
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
      if (hidden.columns.has(col)) continue;
      const type = c.getAttribute("t");
      const raw = first(c, "v")?.textContent ?? "";
      const formula = first(c, "f")?.textContent ?? "";
      const style = styles[Number(c.getAttribute("s") ?? 0)] ?? {};
      let text = raw;
      let numeric = false;
      // A formula with no cached result, as openpyxl and similar writers save them.
      if (raw === "" && formula) text = `=${formula}`;
      else if (type === "s") text = strings[Number(raw)] ?? "";
      else if (type === "inlineStr") text = inlineText(c);
      else if (type === "b") text = raw === "1" ? "TRUE" : "FALSE";
      else if (type !== "str" && type !== "e" && raw !== "" && Number.isFinite(Number(raw))) {
        text = formatNumber(Number(raw), style.format, date1904);
        numeric = true;
      }
      if (text === "" && !style.bold) continue;
      cells[col] = { text, numeric, bold: style.bold, italic: style.italic };
      budget.cells--;
    }
    rows[r] = cells;
  }
  return { name, rows, widths, truncated, hidden };
}

export function readXlsx(bytes: Uint8Array): Sheet[] {
  const files = unpack(bytes, (name) => name.endsWith(".xml") || name.endsWith(".rels"));
  const workbook = xml(files, "xl/workbook.xml");
  if (!workbook) throw new Error("Not a valid XLSX workbook.");
  const rels = relationships(files, "xl/workbook.xml");
  const sharedDoc = xml(files, "xl/sharedStrings.xml");
  const strings = sharedDoc ? all(sharedDoc, "si").map(inlineText) : [];
  const styles = readStyles(xml(files, "xl/styles.xml"));
  const date1904 = ["1", "true"].includes(first(workbook, "workbookPr")?.getAttribute("date1904") ?? "");
  const sheets: Sheet[] = [];
  const budget = { cells: MAX_WORKBOOK_CELLS };
  for (const sheet of all(workbook, "sheet")) {
    if (sheet.getAttribute("state") === "hidden" || sheet.getAttribute("state") === "veryHidden") continue;
    if (budget.cells <= 0) {
      // The sheets after this one are left out.
      const last = sheets.at(-1);
      if (last) last.truncated = true;
      break;
    }
    const path = rels.get(relId(sheet, "id") ?? "");
    const doc = path ? xml(files, path) : null;
    if (doc) sheets.push(readSheet(doc, sheet.getAttribute("name") ?? "Sheet", strings, styles, date1904, budget));
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
  /** A table's cell text, by row. */
  table?: string[][];
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

/** `images: false` reads text only, leaving slide media unpacked. */
export function readPptx(bytes: Uint8Array, { images = true } = {}): Deck {
  const files = unpack(
    bytes,
    (name) =>
      (name.startsWith("ppt/") && (images || /\.(xml|rels)$/.test(name))) || name.startsWith("_rels/"),
  );
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
    // A hidden slide is left out of the show, so out of the viewer and the model's text too.
    if (!path || !doc || ["0", "false"].includes(doc.documentElement.getAttribute("show") ?? "")) continue;
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
    // Tables sit in a graphicFrame, not a shape.
    for (const frame of all(doc, "graphicFrame")) {
      const tbl = first(frame, "tbl");
      if (!tbl) continue;
      const table = children(tbl, "tr").map((tr) =>
        children(tr, "tc").map((tc) =>
          all(tc, "p")
            .map((p) => all(p, "t").map((t) => t.textContent ?? "").join(""))
            .join("\n"),
        ),
      );
      if (!table.some((row) => row.some((cell) => cell.trim()))) continue;
      boxes.push({ frame: readFrame(frame, cx, cy) ?? { x: 0.05, y: 0.25, w: 0.9, h: 0.65 }, table });
    }
    for (const pic of images ? all(doc, "pic") : []) {
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
