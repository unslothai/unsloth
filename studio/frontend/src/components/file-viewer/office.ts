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

/** Inflates the parts `wanted` picks, on demand. The unpacked total counts across calls. */
function archive(bytes: Uint8Array): (wanted: (name: string) => boolean) => Unzipped {
  let total = 0;
  return (wanted) =>
    unzipSync(bytes, {
      filter: (entry) => {
        if (!wanted(entry.name)) return false;
        total += entry.originalSize;
        if (total > MAX_UNPACKED_BYTES) throw new Error("File is too large to preview.");
        return true;
      },
    });
}

// A deck's text parts. Pictures are inflated only for the viewer.
const PPTX_TEXT_PARTS = (name: string) =>
  (name.startsWith("ppt/") && /\.(xml|rels)$/.test(name)) || name.startsWith("_rels/");

function xml(files: Unzipped, path: string): Document | null {
  const bytes = files[path];
  return bytes ? parseXml(strFromU8(bytes)) : null;
}

function parseXml(text: string): Document | null {
  const doc = new DOMParser().parseFromString(text, "application/xml");
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

/** A part's relationships, targets resolved against the part. */
function relationshipList(files: Unzipped, part: string): { id: string; type: string; path: string }[] {
  const slash = part.lastIndexOf("/");
  const dir = part.slice(0, slash + 1);
  const doc = xml(files, relsPath(part));
  const out: { id: string; type: string; path: string }[] = [];
  for (const rel of doc ? all(doc, "Relationship") : []) {
    const id = rel.getAttribute("Id");
    if (!id || rel.getAttribute("TargetMode") === "External") continue;
    out.push({ id, type: rel.getAttribute("Type") ?? "", path: resolvePath(dir, rel.getAttribute("Target") ?? "") });
  }
  return out;
}

/** Relationship id to part path. */
function relationships(files: Unzipped, part: string): Map<string, string> {
  return new Map(relationshipList(files, part).map((rel) => [rel.id, rel.path]));
}

function relsPath(part: string): string {
  const slash = part.lastIndexOf("/");
  return `${part.slice(0, slash + 1)}_rels/${part.slice(slash + 1)}.rels`;
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

/** Memoized by format code: a sheet shares a handful of formats. */
function memo<T>(compute: (code: string) => T): (code: string) => T {
  const cache = new Map<string, T>();
  return (code) => {
    let value = cache.get(code);
    if (value === undefined) {
      // Bounded, as it outlives each file.
      if (cache.size >= 1000) cache.clear();
      value = compute(code);
      cache.set(code, value);
    }
    return value;
  };
}

const dateTokens = memo((code) => {
  const tokens = code.match(DATE_TOKEN) ?? [];
  return {
    tokens,
    kinds: tokens.map((token) => (/^[ymdhs]/i.test(token) && !token.startsWith("\\") ? token[0]!.toLowerCase() : "")),
    twelve: tokens.some((token) => /^(am\/pm|a\/p)$/i.test(token)),
    fraction: /s\.(0+)/i.exec(code)?.[1]?.length ?? 0,
  };
});

/** A date or time, token by token as the format writes it. `m` is minutes after an hour or before
 *  a second, otherwise the month. */
function formatDate(serial: number, code: string): string {
  const { tokens, kinds, twelve, fraction } = dateTokens(code);
  const unit = 1000 / 10 ** fraction;
  // The 1900 system counts a 29 February 1900 that never was (serial 60), so earlier serials run a day ahead.
  const whole = Math.floor(serial);
  const shifted = serial < 60 ? serial + 1 : serial;
  const date = new Date(Math.round(((shifted - 25569) * 86400000) / unit) * unit);
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
const fractionShape = memo((code) => /(?:([0#?]+)\s+)?[0#?]+\s*\/\s*([1-9]\d*|[0#?]+)/.exec(code));

function formatFraction(value: number, code: string): string | null {
  const match = fractionShape(code);
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
  // Fractional seconds (ss.00) at their own precision.
  const digits = /\.(0+)/.exec(code)?.[1]?.length ?? 0;
  const ticks = Math.round(Math.abs(value) * 86400 * 10 ** digits);
  const total = Math.floor(ticks / 10 ** digits);
  const unit = /\[h+\]/i.test(code) ? 3600 : /\[m+\]/i.test(code) ? 60 : 1;
  const lead = Math.floor(total / unit);
  const pad = (n: number) => String(n).padStart(2, "0");
  const out = code
    .replace(/\.0+/, `.${String(ticks % 10 ** digits).padStart(digits, "0")}`)
    .replace(/\[[hms]+\]/i, String(lead))
    .replace(/m+/i, pad(Math.floor((total % 3600) / 60)))
    .replace(/s+/i, pad(total % 60));
  return value < 0 ? `-${out}` : out;
}

function generalText(value: number): string {
  // 15 significant digits, as Excel stores: drops binary noise such as 0.30000000000000004.
  return String(Number.isInteger(value) ? value : Number(value.toPrecision(15)));
}

const isPlaceholder = (token: string) => token === "0" || token === "#" || token === "?";
// What an unused placeholder shows: 0 a zero, ? a space, # nothing.
const emptyPlaceholder = (token: string) => (token === "0" ? "0" : token === "?" ? " " : "");

function literalText(token: string): string {
  if (token.startsWith('"')) return token.slice(1, -1);
  if (token.startsWith("\\")) return token.slice(1);
  return token === "," ? "" : token;
}

/** A non-negative number set into a format's digit placeholders, literals kept in place:
 *  00000 pads a ZIP code, 0.## drops trailing zeros, 000-00-0000 reads as a mask. */
/** A format's digit placeholders and literals, parsed once. */
const digitLayout = memo((format) => {
  const tokens: string[] = format.match(/"[^"]*"|\\.|[0#?.,]|[^"\\0#?.,]+/g) ?? [];
  const dot = tokens.indexOf(".");
  const whole = dot === -1 ? tokens : tokens.slice(0, dot);
  const part = dot === -1 ? [] : tokens.slice(dot + 1);
  const first = whole.findIndex(isPlaceholder);
  const last = whole.length - 1 - [...whole].reverse().findIndex(isPlaceholder);
  return {
    dot,
    whole,
    part,
    first,
    places: part.filter(isPlaceholder),
    // Grouped (#,##0): a comma between placeholders.
    grouped: whole.some((token, i) => token === "," && i > first && i < last),
    zeros: whole.filter((token) => token === "0").length,
    before: whole.slice(0, first).map(literalText).join(""),
    after: whole.slice(last + 1).map(literalText).join(""),
  };
});

function placeDigits(number: number, format: string): string {
  const { dot, whole, part, first, places, grouped, zeros, before, after } = digitLayout(format);
  const [intDigits = "", fracDigits = ""] = number.toFixed(places.length).split(".");
  if (/e/i.test(intDigits)) return generalText(number);
  // No leading zero of its own: 0.5 in #.## shows .5.
  const significant = intDigits === "0" ? "" : intDigits;
  let integer: string;
  if (grouped) {
    // The digits as one run, padded to the zeros the format asks for.
    integer = `${before}${significant.padStart(zeros, "0").replace(/\B(?=(\d{3})+$)/g, ",")}${after}`;
  } else {
    // Right to left, one digit a placeholder; the first takes whatever is left over.
    let left = significant;
    const out = whole.map(() => "");
    for (let i = whole.length - 1; i >= 0; i--) {
      const token = whole[i]!;
      if (!isPlaceholder(token)) out[i] = literalText(token);
      else if (i === first) out[i] = left || emptyPlaceholder(token);
      else out[i] = left.slice(-1) || emptyPlaceholder(token);
      if (isPlaceholder(token)) left = i === first ? "" : left.slice(0, -1);
    }
    integer = out.join("");
  }
  if (dot === -1) return integer;
  const digits = fracDigits.split("");
  // Trailing zeros: a # drops its own, a ? leaves a space.
  for (let i = places.length - 1; i >= 0 && digits[i] === "0" && places[i] !== "0"; i--) {
    digits[i] = emptyPlaceholder(places[i]!);
  }
  let next = 0;
  return `${integer}.${part.map((token) => (isPlaceholder(token) ? digits[next++] : literalText(token))).join("")}`;
}

const CONDITION = /\[(<=|>=|<>|<|>|=)\s*(-?\d+(?:\.\d+)?)\]/;

const conditionOf = memo((section) => CONDITION.exec(section));

function meets(value: number, section: string | undefined): boolean | null {
  const match = section ? conditionOf(section) : null;
  if (!match) return null;
  const limit = Number(match[2]);
  switch (match[1]) {
    case "<":
      return value < limit;
    case ">":
      return value > limit;
    case "<=":
      return value <= limit;
    case ">=":
      return value >= limit;
    case "=":
      return value === limit;
    default:
      return value !== limit;
  }
}

/** Which section formats `value`, and whether a minus goes in front. Plain sections are
 *  positive;negative;zero, a negative section writing its own sign. With conditions ([>=100]),
 *  the first section whose condition holds, the section after them catching the rest. */
function pickSection(value: number, sections: string[]): { index: number; sign: string } {
  const first = meets(value, sections[0]);
  const second = meets(value, sections[1]);
  if (first === null && second === null) {
    const index = value < 0 && sections.length > 1 ? 1 : value === 0 && sections.length > 2 ? 2 : 0;
    return { index, sign: value < 0 && index === 0 ? "-" : "" };
  }
  // A conditional section shows the value's own sign.
  const sign = value < 0 ? "-" : "";
  if (first) return { index: 0, sign };
  if (second || (second === null && sections.length > 1)) return { index: 1, sign };
  return { index: Math.min(2, sections.length - 1), sign };
}

/** The common shapes of an Excel number format: digit placeholders, grouping, percent, currency,
 *  fractions, dates. `date1904`: the workbook counts dates from 1904, 1,462 days after the 1900 system. */
/** A format's sections, split at semicolons outside quotes: `0.00 "kg; net"` is one. */
const splitSections = memo((code) => {
  const sections: string[] = [];
  let start = 0;
  for (let i = 0; i < code.length; i++) {
    const char = code[i];
    if (char === '"') {
      const close = code.indexOf('"', i + 1);
      i = close === -1 ? code.length : close;
    } else if (char === "\\") i++;
    else if (char === ";") {
      sections.push(code.slice(start, i));
      start = i + 1;
    }
  }
  sections.push(code.slice(start));
  return sections;
});

type SectionKind = "elapsed" | "date" | "literal" | "scientific" | "number";

/** A section's kind, with tags dropped (`tagged`) and quotes undone (`code`). */
const readSection = memo((section) => {
  // [$€-407]-style currency tags keep their symbol; [Red] and the like go, but not the
  // elapsed-time units [h], [m] and [s]. Padding (_x) and fill (*x) go too.
  const tagged = section
    .replace(/\[\$([^\]-]*)[^\]]*\]/g, "$1")
    .replace(/\[(?![hms]+\])[^\]]*\]/gi, "")
    .replace(/_.|\*./g, "");
  // Quoted text and escapes are literal: kept in `code`, left out of `bare`, which says what the format is.
  const code = tagged.replace(/"([^"]*)"/g, "$1").replace(/\\(.)/g, "$1");
  const bare = tagged.replace(/"[^"]*"|\\./g, "");
  // Fractional seconds (ss.0, [ss].0) are a time, not a number.
  const dateLike = !/[0#?]/.test(bare.replace(/s\]?\.0+/gi, "s"));
  const kind: SectionKind =
    dateLike && /\[[hms]+\]/i.test(bare)
      ? "elapsed"
      : dateLike && /[dmyhs]/i.test(bare.replace(/general/gi, ""))
        ? "date"
        : !/[0#?]/.test(bare)
          ? "literal"
          : /E[+-]/i.test(bare)
            ? "scientific"
            : "number";
  // Commas after the last digit placeholder scale by a thousand each: #,##0,, shows millions.
  const scale = 1000 ** (bare.match(/[0#?](,+)(?:\.|[^0#?]*$)/)?.[1]?.length ?? 0);
  return { tagged, code, kind, percent: bare.includes("%"), scale };
});

/** Where a format's exponent (E+ or E-) starts, outside quoted text; -1 when it has none. */
function exponentAt(format: string): number {
  for (let i = 0; i < format.length; i++) {
    const char = format[i];
    if (char === '"') {
      const close = format.indexOf('"', i + 1);
      i = close === -1 ? format.length : close;
    } else if (char === "\\") i++;
    else if ((char === "E" || char === "e") && (format[i + 1] === "+" || format[i + 1] === "-")) return i;
  }
  return -1;
}

/** Scientific form (1.23E+03), literals kept (`0.00E+00 "kg"`). E- shows only a negative sign. */
function formatScientific(value: number, format: string): string {
  const at = exponentAt(format);
  const mantissaFormat = format.slice(0, at);
  const exponentFormat = format.slice(at + 2);
  const tokens: string[] = mantissaFormat.match(/"[^"]*"|\\.|[0#?.]|[^"\\0#?.]+/g) ?? [];
  const dot = tokens.indexOf(".");
  const places = dot === -1 ? 0 : tokens.slice(dot + 1).filter(isPlaceholder).length;
  const [mantissa = "0", exponentText = "0"] = value.toExponential(places).split("e");
  const exponent = Number(exponentText);
  const expSign = exponent < 0 ? "-" : format[at + 1] === "+" ? "+" : "";
  return `${placeDigits(Number(mantissa), mantissaFormat)}E${expSign}${placeDigits(Math.abs(exponent), exponentFormat)}`;
}

export function formatNumber(value: number, rawCode: string | undefined, date1904 = false): string {
  if (!rawCode || rawCode === "General" || rawCode === "@") return generalText(value);
  const sections = splitSections(rawCode);
  const { index, sign } = pickSection(value, sections);
  const { tagged, code, kind, percent, scale } = readSection(sections[index] ?? rawCode);
  if (kind === "elapsed") return formatElapsed(value, code.toLowerCase());
  if (kind === "date") return formatDate(date1904 ? value + 1462 : value, tagged);
  // No digit placeholders: literal text, with the value wherever "General" stands.
  if (kind === "literal") return `${sign}${code.replace(/general/i, generalText(Math.abs(value)))}`.trim();
  if (kind === "scientific") return `${sign}${formatScientific(Math.abs(value), tagged).trim()}`;
  const fraction = formatFraction(value, code);
  if (fraction !== null) return fraction;
  const number = Math.abs(percent ? value * 100 : value) / scale;
  return `${sign}${placeDigits(number, tagged).trim()}`;
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

/** A shared formula moved from its master cell to one `rows` and `columns` away: relative
 *  references shift, $-anchored parts and quoted text stay. */
function shiftFormula(formula: string, rows: number, columns: number): string {
  return formula
    .split(/("[^"]*")/)
    .map((part, index) =>
      index % 2
        ? part
        : part.replace(
            /(^|[^A-Za-z0-9_.$])(\$?)([A-Z]{1,3})(\$?)(\d+)(?![\d(A-Za-z_!])/g,
            (_, lead: string, colAbs: string, col: string, rowAbs: string, row: string) =>
              `${lead}${colAbs}${colAbs ? col : columnName(columnIndex(col) + columns)}${rowAbs}${rowAbs ? row : Number(row) + rows}`,
          ),
    )
    .join("");
}

// ---------------------------------------------------------------------------------------------
// Worksheet XML, read as text: several times faster than DOMParser on large sheets, with no DOM.

const NAMED_ENTITIES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };
const ENTITY = /&(?:#x([0-9a-f]+)|#(\d+)|(amp|lt|gt|quot|apos));|<!\[CDATA\[([\s\S]*?)\]\]>/gi;

/** Character references and CDATA undone, as textContent reads them. */
function decodeXml(text: string): string {
  if (!text.includes("&") && !text.includes("<!")) return text;
  return text.replace(ENTITY, (match, hex?: string, dec?: string, name?: string, cdata?: string) => {
    if (cdata !== undefined) return cdata;
    if (name) return NAMED_ENTITIES[name.toLowerCase()]!;
    const code = hex ? parseInt(hex, 16) : Number(dec);
    return code <= 0x10ffff ? String.fromCodePoint(code) : match;
  });
}

const attributePatterns = new Map<string, RegExp>();

/** An attribute's value from a tag's attribute text; null when it has none. */
function attribute(attributes: string, name: string): string | null {
  let pattern = attributePatterns.get(name);
  if (!pattern) {
    pattern = new RegExp(`(?:^|\\s)${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`);
    attributePatterns.set(name, pattern);
  }
  const match = pattern.exec(attributes);
  return match ? decodeXml(match[1] ?? match[2] ?? "") : null;
}

/** An element's start tag, any prefix. */
const openTag = (name: string) => new RegExp(`<(?:[\\w.-]+:)?${name}(?=[\\s/>])([^>]*?)(/)?>`, "g");
const closeTag = (name: string) => new RegExp(`</(?:[\\w.-]+:)?${name}\\s*>`, "g");

/** Each element as [attributes, inner text]. An unclosed element ends the scan, which would
 *  otherwise go quadratic. */
function* elements(text: string, open: RegExp, close: RegExp): Generator<[string, string]> {
  open.lastIndex = 0;
  for (let match = open.exec(text); match; match = open.exec(text)) {
    if (match[2]) {
      yield [match[1]!, ""];
      continue;
    }
    close.lastIndex = open.lastIndex;
    const end = close.exec(text);
    if (!end) return;
    const inner = text.slice(open.lastIndex, end.index);
    open.lastIndex = close.lastIndex;
    yield [match[1]!, inner];
  }
}

const ROW_OPEN = openTag("row");
const ROW_CLOSE = closeTag("row");
const CELL_OPEN = openTag("c");
const CELL_CLOSE = closeTag("c");
const COL_OPEN = openTag("col");
const COL_CLOSE = closeTag("col");
const VALUE = /<(?:[\w.-]+:)?v(?:\s[^>]*)?>([\s\S]*?)<\/(?:[\w.-]+:)?v\s*>/;
const FORMULA = /<(?:[\w.-]+:)?f(?=[\s/>])([^>]*?)(?:\/>|>([\s\S]*?)<\/(?:[\w.-]+:)?f\s*>)/;
const PHONETIC = /<(?:[\w.-]+:)?rPh(?=[\s/>])[^>]*?(?:\/>|>[\s\S]*?<\/(?:[\w.-]+:)?rPh\s*>)/g;
const TEXT_RUN = /<(?:[\w.-]+:)?t(?:\s[^>]*)?>([\s\S]*?)<\/(?:[\w.-]+:)?t\s*>/g;
const SHEET_DATA = /<(?:[\w.-]+:)?sheetData[\s/>]/;

/** A string's text (<si>, <is>), runs joined, phonetic runs (<rPh>) dropped. */
function stringText(xmlText: string): string {
  let out = "";
  for (const match of xmlText.replace(PHONETIC, "").matchAll(TEXT_RUN)) out += decodeXml(match[1]!);
  return out;
}

const isTrue = (value: string | null) => value === "1" || value === "true";

function readSheet(
  text: string,
  name: string,
  string: (index: number) => string,
  styles: CellStyle[],
  date1904: boolean,
  budget: { cells: number },
): Sheet {
  const rows: (SheetCell | undefined)[][] = [];
  const widths: (number | undefined)[] = [];
  const hidden = { rows: new Set<number>(), columns: new Set<number>() };
  const dataAt = text.search(SHEET_DATA);
  const head = dataAt === -1 ? text : text.slice(0, dataAt);
  for (const [attrs] of elements(head, COL_OPEN, COL_CLOSE)) {
    const min = Number(attribute(attrs, "min") ?? 1);
    const max = Math.min(Number(attribute(attrs, "max") ?? min), MAX_SHEET_COLUMNS);
    const width = Number(attribute(attrs, "width"));
    const hide = isTrue(attribute(attrs, "hidden"));
    for (let i = min; i <= max; i++) {
      if (hide) hidden.columns.add(i - 1);
      else if (width) widths[i - 1] = Math.round(width * 7 + 5);
    }
  }
  if (dataAt === -1) return { name, rows, widths, truncated: false, hidden };
  const data = text.slice(dataAt);
  // Shared formulas: the master cell holds the text and comes before the cells sharing it.
  const shared = new Map<string, { formula: string; row: number; col: number }>();
  let truncated = false;
  let nextRow = 0;
  for (const [rowAttrs, rowBody] of elements(data, ROW_OPEN, ROW_CLOSE)) {
    const r = Number(attribute(rowAttrs, "r") ?? nextRow + 1) - 1;
    nextRow = r + 1;
    if (r >= MAX_SHEET_ROWS || budget.cells <= 0) {
      truncated = true;
      break;
    }
    budget.cells--;
    // Hidden, as Excel shows it: neither in the grid nor in the text sent to the model.
    if (isTrue(attribute(rowAttrs, "hidden"))) {
      hidden.rows.add(r);
      continue;
    }
    const cells: (SheetCell | undefined)[] = [];
    let nextColumn = 0;
    for (const [attrs, body] of elements(rowBody, CELL_OPEN, CELL_CLOSE)) {
      const ref = attribute(attrs, "r");
      const col = ref ? columnIndex(ref) : nextColumn;
      nextColumn = col + 1;
      if (col >= MAX_SHEET_COLUMNS) {
        truncated = true;
        continue;
      }
      if (hidden.columns.has(col)) continue;
      const type = attribute(attrs, "t");
      const raw = decodeXml(VALUE.exec(body)?.[1] ?? "");
      // Cheap check first: most cells have no formula.
      const f = body.includes("f") ? FORMULA.exec(body) : null;
      const fAttrs = f?.[1] ?? "";
      const fText = decodeXml(f?.[2] ?? "");
      const sharedId = f && attribute(fAttrs, "t") === "shared" ? (attribute(fAttrs, "si") ?? "") : null;
      if (sharedId !== null && fText && ref) {
        shared.set(sharedId, { formula: fText, row: r, col });
      }
      const master = sharedId !== null ? shared.get(sharedId) : undefined;
      const formula = fText || (master ? shiftFormula(master.formula, r - master.row, col - master.col) : "");
      const style = styles[Number(attribute(attrs, "s") ?? 0)] ?? {};
      let value = raw;
      let numeric = false;
      // A formula with no cached result, as openpyxl and similar writers save them.
      if (raw === "" && formula) value = `=${formula}`;
      else if (type === "s") value = string(Number(raw));
      else if (type === "inlineStr") value = stringText(body);
      else if (type === "b") value = raw === "1" ? "TRUE" : "FALSE";
      else if (type !== "str" && type !== "e" && raw !== "" && Number.isFinite(Number(raw))) {
        value = formatNumber(Number(raw), style.format, date1904);
        numeric = true;
      }
      if (value === "" && !style.bold) continue;
      cells[col] = { text: value, numeric, bold: style.bold, italic: style.italic };
      budget.cells--;
    }
    rows[r] = cells;
  }
  return { name, rows, widths, truncated, hidden };
}

// Sheet parts past this are decoded only up to the rows kept.
const SHEET_CUT_BYTES = 1024 * 1024;
// Upper bound on sheet XML read, since a row may hold 16,384 cells.
const MAX_SHEET_XML_BYTES = 64 * 1024 * 1024;

/** Byte offset after the `rows`th `</row>` (any prefix), or after the last one within
 *  MAX_SHEET_XML_BYTES; -1 if the part ends first. Scans bytes, so the rest is never decoded. */
function rowsEnd(bytes: Uint8Array, rows: number): { end: number; capped: boolean } {
  let count = 0;
  let last = -1;
  for (let i = bytes.indexOf(0x3c); i !== -1; i = bytes.indexOf(0x3c, i + 1)) {
    if (i > MAX_SHEET_XML_BYTES) return { end: last, capped: true };
    if (bytes[i + 1] !== 0x2f) continue;
    // The local name, after a prefix if the writer gave one.
    let name = i + 2;
    for (let j = name; j < i + 40 && bytes[j] !== 0x3e; j++) {
      if (bytes[j] === 0x3a) {
        name = j + 1;
        break;
      }
    }
    if (bytes[name] === 0x72 && bytes[name + 1] === 0x6f && bytes[name + 2] === 0x77 && bytes[name + 3] === 0x3e) {
      last = name + 4;
      if (++count === rows) return { end: last, capped: false };
    }
  }
  return { end: -1, capped: false };
}

/** A worksheet's text up to its first `rows` rows. `cut` when rows were left out. */
function sheetText(bytes: Uint8Array, rows: number): { text: string; cut: boolean } {
  if (bytes.length <= SHEET_CUT_BYTES) return { text: strFromU8(bytes), cut: false };
  const { end, capped } = rowsEnd(bytes, rows);
  if (end === -1) return { text: strFromU8(bytes), cut: false };
  const more = capped || rowsEnd(bytes.subarray(end), 1).end !== -1;
  return { text: strFromU8(bytes.subarray(0, end)), cut: more };
}

const SHARED_STRING_OPEN = openTag("si");
const SHARED_STRING_CLOSE = closeTag("si");

export function readXlsx(bytes: Uint8Array): Sheet[] {
  const read = archive(bytes);
  const main = "xl/workbook.xml";
  const head = read((name) => name === main || name === relsPath(main));
  const workbook = xml(head, main);
  if (!workbook) throw new Error("Not a valid XLSX workbook.");
  // Only parts the workbook uses: not custom XML, pivot caches or drawings.
  const rels = relationshipList(head, main);
  const partOf = (type: string, fallback: string) => rels.find((rel) => rel.type.endsWith(`/${type}`))?.path ?? fallback;
  const stringsPath = partOf("sharedStrings", "xl/sharedStrings.xml");
  const stylesPath = partOf("styles", "xl/styles.xml");
  const support = read((name) => name === stringsPath || name === stylesPath);
  // Shared strings are decoded lazily, as cells use them.
  const sharedPart = support[stringsPath];
  const items = sharedPart
    ? Array.from(elements(strFromU8(sharedPart), SHARED_STRING_OPEN, SHARED_STRING_CLOSE), ([, body]) => body)
    : [];
  const strings: (string | undefined)[] = [];
  const string = (index: number) => {
    const item = items[index];
    return item === undefined ? "" : (strings[index] ??= stringText(item));
  };
  const styles = readStyles(xml(support, stylesPath));
  const date1904 = ["1", "true"].includes(first(workbook, "workbookPr")?.getAttribute("date1904") ?? "");
  const paths = new Map(rels.map((rel) => [rel.id, rel.path]));
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
    const path = paths.get(relId(sheet, "id") ?? "");
    const part = path ? read((name) => name === path)[path] : undefined;
    if (!part) continue;
    const { text, cut } = sheetText(part, Math.min(MAX_SHEET_ROWS, budget.cells));
    const parsed = readSheet(text, sheet.getAttribute("name") ?? "Sheet", string, styles, date1904, budget);
    if (cut) parsed.truncated = true;
    sheets.push(parsed);
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
  /** A picture's bytes. The viewer makes a URL only while its slide is mounted. */
  image?: Blob;
  /** A table's cell text, by row. A chart's cached data comes as one too. */
  table?: string[][];
  /** A chart's title, shown above its data. */
  caption?: string;
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

const imageType = (path: string) => {
  const extension = path.split(".").pop()!.toLowerCase();
  return Object.hasOwn(IMAGE_TYPES, extension) ? IMAGE_TYPES[extension] : undefined;
};

// A chart shown as a table, capped so a chart of many long series stays a readable size.
const MAX_CHART_SERIES = 100;
const MAX_CHART_CELLS = 5000;

/** A table's cell text, capped like a chart's. */
function readTable(tbl: Element): string[][] {
  const table: string[][] = [];
  let cells = 0;
  let cut = false;
  for (const tr of children(tbl, "tr")) {
    const tcs = children(tr, "tc");
    if (tcs.length > MAX_CHART_SERIES) cut = true;
    const row = tcs.slice(0, MAX_CHART_SERIES);
    if (cells + row.length > MAX_CHART_CELLS) {
      cut = true;
      break;
    }
    cells += row.length;
    table.push(row.map((tc) => all(tc, "p").map(paragraphText).join("\n")));
  }
  if (cut) table.push(["…"]);
  return table;
}

const MAX_DIAGRAM_NODES = 500;

/** A SmartArt diagram's node labels, from its data part. */
function readDiagram(doc: Document): NonNullable<SlideBox["paragraphs"]> {
  const paragraphs: NonNullable<SlideBox["paragraphs"]> = [];
  for (const pt of all(doc, "pt")) {
    // Nodes only: transitions and connections carry no text.
    const type = pt.getAttribute("type");
    if (type && type !== "node") continue;
    const text = all(pt, "p").map(paragraphText).join("\n");
    if (text.trim()) paragraphs.push({ text, bullet: true });
    if (paragraphs.length >= MAX_DIAGRAM_NODES) break;
  }
  return paragraphs;
}

/** A chart's cached data as a table: a header of series names, then one row a category. A last
 *  row of "…" marks data left out. */
function readChart(doc: Document): { caption?: string; table: string[][] } | null {
  const serNodes = all(doc, "ser");
  const limit = Math.floor(MAX_CHART_CELLS / (Math.min(serNodes.length, MAX_CHART_SERIES) + 1));
  let cut = serNodes.length > MAX_CHART_SERIES;
  const cache = (node: Element | undefined) => {
    const out: string[] = [];
    for (const pt of node ? all(node, "pt") : []) {
      const idx = Number(pt.getAttribute("idx") ?? out.length);
      if (idx >= limit) cut = true;
      else if (idx >= 0) out[idx] = first(pt, "v")?.textContent ?? "";
    }
    return out;
  };
  const series = serNodes.slice(0, MAX_CHART_SERIES).map((ser) => {
    const tx = children(ser, "tx")[0];
    return {
      name: (tx && (cache(tx)[0] ?? first(tx, "v")?.textContent)) ?? "",
      categories: cache(children(ser, "cat")[0] ?? children(ser, "xVal")[0]),
      values: cache(children(ser, "val")[0] ?? children(ser, "yVal")[0]),
    };
  });
  if (!series.length) return null;
  const categories = series.find((s) => s.categories.length)?.categories ?? [];
  const count = series.reduce((n, s) => Math.max(n, s.values.length), categories.length);
  const table = [["", ...series.map((s) => s.name)]];
  for (let i = 0; i < count; i++) table.push([categories[i] ?? String(i + 1), ...series.map((s) => s.values[i] ?? "")]);
  if (cut) table.push(["…"]);
  const title = first(doc, "title");
  const caption = title ? all(title, "t").map((t) => t.textContent ?? "").join("") : "";
  return { caption: caption || undefined, table };
}

/** A paragraph's text in order, a manual line break (<a:br/>) kept as a newline. */
function paragraphText(p: Element): string {
  return Array.from(p.children)
    .map((child) =>
      child.localName === "br"
        ? "\n"
        : child.localName === "r" || child.localName === "fld"
          ? all(child, "t").map((t) => t.textContent ?? "").join("")
          : "",
    )
    .join("");
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
  // Pictures only, not video or audio.
  const files = archive(bytes)(
    (name) => (images && name.startsWith("ppt/") && imageType(name) !== undefined) || PPTX_TEXT_PARTS(name),
  );
  const presentation = xml(files, "ppt/presentation.xml");
  if (!presentation) throw new Error("Not a valid PPTX presentation.");
  const size = first(presentation, "sldSz");
  const cx = Number(size?.getAttribute("cx")) || 12192000;
  const cy = Number(size?.getAttribute("cy")) || 6858000;
  const rels = relationships(files, "ppt/presentation.xml");
  const slides: Slide[] = [];
  // One Blob per picture, however many slides use it.
  const pictures = new Map<string, Blob>();
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
            text: paragraphText(p),
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
    // Tables, charts and SmartArt sit in a graphicFrame. A chart shows its cached data.
    for (const frame of all(doc, "graphicFrame")) {
      const place = () => readFrame(frame, cx, cy) ?? { x: 0.05, y: 0.25, w: 0.9, h: 0.65 };
      const chartRef = first(frame, "chart");
      const chartPath = chartRef && slideRels.get(relId(chartRef, "id") ?? "");
      const chartDoc = chartPath ? xml(files, chartPath) : null;
      const chart = chartDoc && readChart(chartDoc);
      if (chart) boxes.push({ frame: place(), ...chart });
      const diagramRef = first(frame, "relIds");
      const diagramPath = diagramRef && slideRels.get(relId(diagramRef, "dm") ?? "");
      const diagramDoc = diagramPath ? xml(files, diagramPath) : null;
      const diagram = diagramDoc ? readDiagram(diagramDoc) : [];
      if (diagram.length) boxes.push({ frame: place(), paragraphs: diagram });
      const tbl = first(frame, "tbl");
      if (!tbl) continue;
      const table = readTable(tbl);
      if (!table.some((row) => row.some((cell) => cell.trim()))) continue;
      boxes.push({ frame: place(), table });
    }
    for (const pic of images ? all(doc, "pic") : []) {
      const blip = first(pic, "blip");
      const target = blip && slideRels.get(relId(blip, "embed") ?? "");
      const data = target && files[target];
      const type = target && imageType(target);
      if (!data || !type) continue;
      const image = pictures.get(target) ?? new Blob([data as Uint8Array<ArrayBuffer>], { type });
      pictures.set(target, image);
      boxes.unshift({ frame: readFrame(pic, cx, cy), image });
    }
    slides.push({ boxes });
  }
  return { aspect: cy / cx, widthPt: cx / 12700, slides };
}
