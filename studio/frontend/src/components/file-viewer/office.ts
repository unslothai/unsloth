// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Unzipped, inflateSync, strFromU8, unzipSync } from "fflate";

/** Readers for the parts of an XLSX or PPTX a viewer shows: values, not a faithful rendering. */

const MAX_UNPACKED_BYTES = 200 * 1024 * 1024;
// A part parsed into a DOM (workbook, presentation, slide, chart, relationships) past this is not
// read, as the ceiling on a Word document's XML: real ones are far smaller.
const MAX_XML_PART_BYTES = 10 * 1024 * 1024;
export const MAX_SHEET_ROWS = 5000;
export const MAX_SHEET_COLUMNS = 200;
// Across every sheet, so a workbook of many sheets reads no more than one full one. Each row also
// costs one, so a full sheet fits.
const MAX_WORKBOOK_CELLS = MAX_SHEET_ROWS * (MAX_SHEET_COLUMNS + 1);
// Visible sheets read: empty ones cost no cells, but each is a tab and an inflate.
const MAX_SHEETS = 100;

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

interface ZipEntry {
  offset: number;
  size: number;
  originalSize: number;
  method: number;
}

/** The central directory, read once: each part's name to where its data sits. Null for ZIP64 or
 *  a directory that does not parse, which unzipSync then reads instead. */
function zipIndex(bytes: Uint8Array, view: DataView): Map<string, ZipEntry> | null {
  let end = bytes.length - 22;
  const stop = Math.max(0, end - 0xffff);
  while (end >= stop && view.getUint32(end, true) !== 0x06054b50) end--;
  if (end < stop) return null;
  const count = view.getUint16(end + 10, true);
  let at = view.getUint32(end + 16, true);
  if (count === 0xffff || at === 0xffffffff) return null;
  const utf8 = new TextDecoder();
  const index = new Map<string, ZipEntry>();
  for (let i = 0; i < count; i++) {
    if (at + 46 > bytes.length || view.getUint32(at, true) !== 0x02014b50) return null;
    const size = view.getUint32(at + 20, true);
    const originalSize = view.getUint32(at + 24, true);
    const offset = view.getUint32(at + 42, true);
    if (size === 0xffffffff || originalSize === 0xffffffff || offset === 0xffffffff) return null;
    const nameEnd = at + 46 + view.getUint16(at + 28, true);
    const raw = bytes.subarray(at + 46, nameEnd);
    // Bit 11 marks a UTF-8 name; otherwise one byte a character, as unzipSync reads it.
    const name = view.getUint16(at + 8, true) & 0x800 ? utf8.decode(raw) : String.fromCharCode(...raw);
    index.set(name, { offset, size, originalSize, method: view.getUint16(at + 10, true) });
    at = nameEnd + view.getUint16(at + 30, true) + view.getUint16(at + 32, true);
  }
  return index;
}

function inflateEntry(bytes: Uint8Array, view: DataView, entry: ZipEntry): Uint8Array<ArrayBuffer> {
  const at = entry.offset;
  if (view.getUint32(at, true) !== 0x04034b50) throw new Error("Not a valid ZIP archive.");
  const start = at + 30 + view.getUint16(at + 26, true) + view.getUint16(at + 28, true);
  const data = bytes.subarray(start, start + entry.size);
  if (entry.method === 0) return data.slice();
  if (entry.method === 8) return inflateSync(data, { out: new Uint8Array(entry.originalSize) });
  throw new Error(`Unsupported ZIP compression method ${entry.method}.`);
}

/** Inflates the named parts, on demand; one past `limit` is left out. The archive is indexed
 *  once, so each read is by lookup, not another pass over every entry. The unpacked total counts
 *  across calls. */
type Reader = ((names: Iterable<string>, limit?: number) => Unzipped) & {
  /** A part's unpacked size, as the archive declares it. */
  size: (name: string) => number | undefined;
};

function archive(bytes: Uint8Array): Reader {
  let total = 0;
  const charge = (size: number) => {
    total += size;
    if (total > MAX_UNPACKED_BYTES) throw new Error("File is too large to preview.");
  };
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const index = zipIndex(bytes, view);
  if (!index) {
    const read = (names: Iterable<string>, limit = Infinity) => {
      const wanted = new Set(names);
      return unzipSync(bytes, {
        filter: (entry) => wanted.has(entry.name) && entry.originalSize <= limit && (charge(entry.originalSize), true),
      });
    };
    // Sizes from one pass over the directory, the first time one is asked for.
    let sizes: Map<string, number> | undefined;
    const size = (name: string) => {
      if (!sizes) {
        const found = new Map<string, number>();
        unzipSync(bytes, { filter: (entry) => (found.set(entry.name, entry.originalSize), false) });
        sizes = found;
      }
      return sizes.get(name);
    };
    return Object.assign(read, { size });
  }
  const read = (names: Iterable<string>, limit = Infinity) => {
    const files: Unzipped = {};
    for (const name of new Set(names)) {
      const entry = index.get(name);
      if (!entry || entry.originalSize > limit) continue;
      charge(entry.originalSize);
      files[name] = inflateEntry(bytes, view, entry);
    }
    return files;
  };
  return Object.assign(read, { size: (name: string) => index.get(name)?.originalSize });
}

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
  // 5 to 8 are locale currency formats, left out of files: the en-US ones.
  5: '"$"#,##0_);("$"#,##0)',
  6: '"$"#,##0_);[Red]("$"#,##0)',
  7: '"$"#,##0.00_);("$"#,##0.00)',
  8: '"$"#,##0.00_);[Red]("$"#,##0.00)',
  1: "0",
  2: "0.00",
  3: "#,##0",
  4: "#,##0.00",
  9: "0%",
  10: "0.00%",
  11: "0.00E+00",
  12: "# ?/?",
  13: "# ??/??",
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
  41: '_(* #,##0_);_(* (#,##0);_(* "-"_);_(@_)',
  42: '_("$"* #,##0_);_("$"* (#,##0);_("$"* "-"_);_(@_)',
  43: '_(* #,##0.00_);_(* (#,##0.00);_(* "-"??_);_(@_)',
  44: '_("$"* #,##0.00_);_("$"* (#,##0.00);_("$"* "-"??_);_(@_)',
  45: "mm:ss",
  46: "[h]:mm:ss",
  47: "mm:ss.0",
  48: "##0.0E+0",
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
function formatDate(serial: number, code: string): string | null {
  const { tokens, kinds, twelve, fraction } = dateTokens(code);
  const unit = 1000 / 10 ** fraction;
  // The 1900 system counts a 29 February 1900 that never was (serial 60), so earlier serials run a day ahead.
  const whole = Math.floor(serial);
  const shifted = serial < 60 ? serial + 1 : serial;
  const date = new Date(Math.round(((shifted - 25569) * 86400000) / unit) * unit);
  if (!Number.isFinite(date.getTime())) return null;
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
const fractionShape = memo((format) =>
  // Quoted and escaped text blanked, same length: 0 "0/0" is a number and a literal.
  /(?:([0#?]+)\s+)?[0#?]+\s*\/\s*([1-9]\d*|[0#?]+)/.exec(format.replace(/"[^"]*"|\\./g, (text) => "\u0001".repeat(text.length))),
);

const unquote = (format: string) => format.replace(/"([^"]*)"/g, "$1").replace(/\\(.)/g, "$1");

/** `format` with its quotes: a literal is never read as placeholders. */
function formatFraction(value: number, format: string): string | null {
  const match = fractionShape(format);
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
  const at = match.index;
  return `${value < 0 ? "-" : ""}${unquote(format.slice(0, at))}${body}${unquote(format.slice(at + text.length))}`.trim();
}

/** An elapsed-time format ([h]:mm:ss, [mm]:ss, [ss]): the bracketed unit counts past its usual range. */
function formatElapsed(value: number, format: string): string {
  // Placeholders only: quoted and escaped text stays as written ([h]:mm "hours").
  const tokens: string[] = format.match(/"[^"]*"|\\.|\[[hms]+\]|h+|m+|s+|\.0+|[\s\S]/gi) ?? [];
  // Fractional seconds (ss.00) at their own precision.
  const digits = tokens.find((token) => /^\.0+$/.test(token))?.length ?? 1;
  const scale = 10 ** (digits - 1);
  const ticks = Math.round(Math.abs(value) * 86400 * scale);
  const total = Math.floor(ticks / scale);
  // As wide as the placeholder: [h]:m:s shows one hour as 1:0:0, [hh]:mm:ss as 01:00:00.
  const pad = (n: number, width: number) => String(n).padStart(Math.min(width, 2), "0");
  const out = tokens
    .map((token) => {
      const unit = token.toLowerCase();
      if (unit.startsWith("[")) {
        const per = unit[1] === "h" ? 3600 : unit[1] === "m" ? 60 : 1;
        return pad(Math.floor(total / per), unit.length - 2);
      }
      if (unit[0] === "h") return pad(Math.floor(total / 3600) % 24, unit.length);
      if (unit[0] === "m") return pad(Math.floor((total % 3600) / 60), unit.length);
      if (unit[0] === "s") return pad(total % 60, unit.length);
      if (/^\.0+$/.test(token)) return `.${String(ticks % scale).padStart(digits - 1, "0")}`;
      if (token.startsWith('"')) return token.slice(1, -1);
      return token.startsWith("\\") ? token.slice(1) : token;
    })
    .join("");
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

const SCALE_COMMAS = /([0#?])(,+)(?=\.|[^0#?]*$)/;

/** A section's kind, with tags dropped (`tagged`) and quotes undone (`code`). */
const readSection = memo((section) => {
  // [$€-407]-style currency tags keep their symbol, quoted; [Red] and the like go, but not the
  // elapsed-time units [h], [m] and [s]. Padding (_x) and fill (*x) go too. Only outside
  // quoted and escaped text: "A_B"0 shows A_B1.
  const tagged = (section.match(/"[^"]*"|\\[\s\S]|\[[^\]]*\]|[_*][\s\S]?|[\s\S]/g) ?? [])
    .map((token) => {
      if (token[0] === "_" || token[0] === "*") return "";
      if (token[0] !== "[" || /^\[[hms]+\]$/i.test(token)) return token;
      const symbol = /^\[\$([^\]-]*)/.exec(token)?.[1];
      return symbol ? `"${symbol}"` : "";
    })
    .join("");
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
  const scale = 1000 ** (bare.match(SCALE_COMMAS)?.[2]?.length ?? 0);
  // Each % scales by a hundred: 0%% shows 0.01 as 100%%.
  return { tagged, code, kind, percents: bare.split("%").length - 1, scale };
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
  // More than one integer placeholder steps the exponent by that many: ##0.0E+0 is engineering
  // notation, 12345 showing as 12.3E+3.
  const step = Math.max(1, (dot === -1 ? tokens : tokens.slice(0, dot)).filter(isPlaceholder).length);
  const [normalized = "0", exponentText = "0"] = value.toExponential(places).split("e");
  let mantissa = normalized;
  let exponent = Number(exponentText);
  if (step > 1 && value !== 0) {
    exponent = Math.floor(Math.log10(value) / step) * step;
    mantissa = (value / 10 ** exponent).toFixed(places);
    // Rounding can carry into the next step: 999.96 in ##0.0E+0 is 1.0E+3.
    if (Number(mantissa) >= 10 ** step) {
      exponent += step;
      mantissa = (value / 10 ** exponent).toFixed(places);
    }
  }
  const expSign = exponent < 0 ? "-" : format[at + 1] === "+" ? "+" : "";
  return `${placeDigits(Number(mantissa), mantissaFormat)}E${expSign}${placeDigits(Math.abs(exponent), exponentFormat)}`;
}

export function formatNumber(value: number, rawCode: string | undefined, date1904 = false): string {
  if (!rawCode || rawCode === "General" || rawCode === "@") return generalText(value);
  const sections = splitSections(rawCode);
  const { index, sign } = pickSection(value, sections);
  const { tagged, code, kind, percents, scale } = readSection(sections[index] ?? rawCode);
  if (kind === "elapsed") return `${sign}${formatElapsed(Math.abs(value), tagged)}`;
  if (kind === "date") return formatDate(date1904 ? value + 1462 : value, tagged) ?? generalText(value);
  // No digit placeholders: literal text, with the value wherever "General" stands.
  if (kind === "literal") return `${sign}${code.replace(/general/i, generalText(Math.abs(value)))}`.trim();
  // The magnitude, percent and thousands scaling applied whatever its shape; the section writes its
  // own sign, as in the other branches. (# ?/?;(# ?/?) shows (1 1/2), # ?/?, shows thousands.)
  const number = (Math.abs(value) * 100 ** percents) / scale;
  // The scaling commas go, or the fraction and exponent would show them as text.
  const unscaled = (format: string) => (scale === 1 ? format : format.replace(SCALE_COMMAS, "$1"));
  if (kind === "scientific") return `${sign}${formatScientific(number, unscaled(tagged)).trim()}`;
  const fraction = formatFraction(number, unscaled(tagged));
  if (fraction !== null) return `${sign}${fraction}`;
  return `${sign}${placeDigits(number, tagged).trim()}`;
}

/** A string cell in its format's text section: the fourth, or a lone section with @.
 *  `0;0;0;"SKU-"@` shows ABC as SKU-ABC. */
function formatText(text: string, rawCode: string | undefined): string {
  if (!rawCode || rawCode === "@" || rawCode === "General") return text;
  const sections = splitSections(rawCode);
  const tokens: string[] = readSection(sections[3] ?? sections[0]!).tagged.match(/"[^"]*"|\\.|[\s\S]/g) ?? [];
  if (sections.length < 4 && (sections.length > 1 || !tokens.includes("@"))) return text;
  return tokens
    .map((token) => (token === "@" ? text : token.startsWith('"') ? token.slice(1, -1) : token.startsWith("\\") ? token.slice(1) : token))
    .join("");
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

/** East Asian built-in dates and times (27 to 36, 50 to 58), which files leave out: shown as the
 *  nearest standard ones rather than as serial numbers. */
function localeFormat(id: number): string | undefined {
  if (id === 32) return "h:mm";
  if (id === 33) return "h:mm:ss";
  return (id >= 27 && id <= 36) || (id >= 50 && id <= 58) ? "m/d/yyyy" : undefined;
}

/** An OOXML on/off flag (<b/>, <b val="0"/>): on unless its val says otherwise. */
const isOn = (flag: Element) => !["0", "false", "off"].includes(flag.getAttribute("val") ?? "");

function readStyles(doc: Document | null): CellStyle[] {
  if (!doc) return [];
  const formats = new Map<number, string>();
  for (const fmt of all(doc, "numFmt")) {
    formats.set(Number(fmt.getAttribute("numFmtId")), fmt.getAttribute("formatCode") ?? "");
  }
  const fontsNode = first(doc, "fonts");
  const fonts = fontsNode
    ? children(fontsNode, "font").map((font) => ({
        bold: children(font, "b").some(isOn),
        italic: children(font, "i").some(isOn),
      }))
    : [];
  const xfs = first(doc, "cellXfs");
  return (xfs ? children(xfs, "xf") : []).map((xf) => {
    const id = Number(xf.getAttribute("numFmtId") ?? 0);
    const font = fonts[Number(xf.getAttribute("fontId") ?? 0)];
    return { format: formats.get(id) ?? BUILTIN_FORMATS[id] ?? localeFormat(id), ...font };
  });
}

// A cell (B2, $b$2), a whole column (A:A) or a whole row (1:1), after anything but a name's letters.
const REFERENCE =
  /(^|[^A-Za-z0-9_.$])(?:(\$?)([A-Za-z]{1,3})(\$?)(\d+)(?![\d(A-Za-z_!])|(\$?)([A-Za-z]{1,3}):(\$?)([A-Za-z]{1,3})(?![\w(!.])|(\$?)(\d+):(\$?)(\d+)(?![\d.]))/g;

/** A shared formula moved from its master cell to one `rows` and `columns` away: relative
 *  references shift, $-anchored parts, quoted text and quoted sheet names stay. */
function shiftFormula(formula: string, rows: number, columns: number): string {
  const column = (abs: string, name: string) =>
    abs + (abs ? name.toUpperCase() : columnName(columnIndex(name.toUpperCase()) + columns));
  const row = (abs: string, n: string) => abs + (abs ? n : String(Number(n) + rows));
  // Quoted text and quoted sheet names ('A1'!B2) are left as they are.
  return formula
    .split(/("(?:[^"]|"")*"|'(?:[^']|'')*')/)
    .map((part, index) =>
      index % 2
        ? part
        : part.replace(REFERENCE, (...m: string[]) => {
            const [, lead, ca, c, ra, r, c1a, c1, c2a, c2, r1a, r1, r2a, r2] = m;
            if (c !== undefined) return `${lead}${column(ca!, c)}${row(ra!, r!)}`;
            if (c1 !== undefined) return `${lead}${column(c1a!, c1)}:${column(c2a!, c2!)}`;
            return `${lead}${row(r1a!, r1!)}:${row(r2a!, r2!)}`;
          }),
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
const VALUE = /<(?:[\w.-]+:)?v(?=[\s/>])[^>]*?(?:\/>|>([\s\S]*?)<\/(?:[\w.-]+:)?v\s*>)/;
const FORMULA = /<(?:[\w.-]+:)?f(?=[\s/>])([^>]*?)(?:\/>|>([\s\S]*?)<\/(?:[\w.-]+:)?f\s*>)/;
const PHONETIC = /<(?:[\w.-]+:)?rPh(?=[\s/>])[^>]*?(?:\/>|>[\s\S]*?<\/(?:[\w.-]+:)?rPh\s*>)/g;
const TEXT_RUN = /<(?:[\w.-]+:)?t(?:\s[^>]*)?>([\s\S]*?)(?:<\/(?:[\w.-]+:)?t\s*>|$)/g;
const SHEET_DATA = /<(?:[\w.-]+:)?sheetData[\s/>]/;

const MARKUP = /<!\[CDATA\[([\s\S]*?)\]\]>|<!--[\s\S]*?-->/g;

/** CDATA as escaped text and comments dropped, so neither can hold a tag the scan would match.
 *  decodeXml undoes the escapes. */
function flattenMarkup(text: string): string {
  if (!text.includes("<!")) return text;
  return text.replace(MARKUP, (_, cdata?: string) =>
    cdata === undefined ? "" : cdata.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"),
  );
}

/** A string's text (<si>, <is>), runs joined, phonetic runs (<rPh>) dropped. */
function stringText(xmlText: string): string {
  let out = "";
  for (const match of flattenMarkup(xmlText).replace(PHONETIC, "").matchAll(TEXT_RUN)) {
    // Past a cell's worth, the rest is never decoded.
    out += decodeXml(match[1]!.slice(0, MAX_CELL_BYTES));
    if (out.length >= MAX_CELL_TEXT) return out.slice(0, MAX_CELL_TEXT);
  }
  return out;
}

const isTrue = (value: string | null) => value === "1" || value === "true";

// Excel's own limit on a cell's text; MAX_CELL_BYTES of XML covers it, entities and all.
const MAX_CELL_TEXT = 32_767;
const MAX_CELL_BYTES = MAX_CELL_TEXT * 10;

function readSheet(
  text: string,
  name: string,
  string: (index: number) => string,
  style: (index: number) => CellStyle | undefined,
  date1904: boolean,
  budget: { cells: number },
): Sheet {
  const rows: (SheetCell | undefined)[][] = [];
  const widths: (number | undefined)[] = [];
  const hidden = { rows: new Set<number>(), columns: new Set<number>() };
  text = flattenMarkup(text);
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
    // Hidden, as Excel shows it: neither in the grid nor in the text sent to the model. Still read
    // for a shared formula's master, which a shown cell below may follow.
    const rowHidden = isTrue(attribute(rowAttrs, "hidden"));
    if (rowHidden) hidden.rows.add(r);
    if (rowHidden && !rowBody.includes("shared")) continue;
    const cells: (SheetCell | undefined)[] = [];
    let nextColumn = 0;
    for (const [attrs, body] of elements(rowBody, CELL_OPEN, CELL_CLOSE)) {
      const ref = attribute(attrs, "r");
      const col = ref ? columnIndex(ref) : nextColumn;
      nextColumn = col + 1;
      if (col >= MAX_SHEET_COLUMNS) {
        truncated ||= !rowHidden;
        continue;
      }
      // Cheap check first: most cells have no formula.
      const f = body.includes("f") ? FORMULA.exec(body) : null;
      const fAttrs = f?.[1] ?? "";
      const fText = decodeXml((f?.[2] ?? "").slice(0, MAX_CELL_BYTES));
      const sharedId = f && attribute(fAttrs, "t") === "shared" ? (attribute(fAttrs, "si") ?? "") : null;
      if (sharedId !== null && fText && ref) {
        shared.set(sharedId, { formula: fText, row: r, col });
      }
      if (rowHidden || hidden.columns.has(col)) continue;
      const type = attribute(attrs, "t");
      const cached = VALUE.exec(body);
      const raw = decodeXml((cached?.[1] ?? "").slice(0, MAX_CELL_BYTES));
      const master = sharedId !== null ? shared.get(sharedId) : undefined;
      const formula = fText || (master ? shiftFormula(master.formula, r - master.row, col - master.col) : "");
      const cellStyle = style(Number(attribute(attrs, "s") ?? 0)) ?? {};
      let value = raw;
      let numeric = false;
      // An empty string cache is a result; an empty numeric cache is uncalculated (openpyxl).
      if (raw === "" && formula && !(type === "str" && cached)) value = `=${formula}`;
      else if (type === "s" || type === "inlineStr" || type === "str") {
        value = formatText(type === "s" ? string(Number(raw)) : type === "inlineStr" ? stringText(body) : raw, cellStyle.format);
      } else if (type === "b") value = isTrue(raw) ? "TRUE" : "FALSE";
      else if (type !== "str" && type !== "e" && raw !== "" && Number.isFinite(Number(raw))) {
        value = formatNumber(Number(raw), cellStyle.format, date1904);
        numeric = true;
      }
      if (value === "" && !cellStyle.bold) continue;
      cells[col] = { text: value.slice(0, MAX_CELL_TEXT), numeric, bold: cellStyle.bold, italic: cellStyle.italic };
      budget.cells--;
    }
    if (!rowHidden) rows[r] = cells;
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
    const skipped = skipMarkup(bytes, i);
    if (skipped === -1) break;
    if (skipped !== i) {
      i = skipped;
      continue;
    }
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
  if (end === -1) {
    // No row closes within the cap: only the head is read, which yields no rows.
    return bytes.length > MAX_SHEET_XML_BYTES || capped
      ? { text: strFromU8(bytes.subarray(0, MAX_SHEET_XML_BYTES)), cut: true }
      : { text: strFromU8(bytes), cut: false };
  }
  const more = capped || rowsEnd(bytes.subarray(end), 1).end !== -1;
  return { text: strFromU8(bytes.subarray(0, end)), cut: more };
}

const CDATA_END = [0x5d, 0x5d, 0x3e]; // ]]>
const COMMENT_END = [0x2d, 0x2d, 0x3e]; // -->

function indexOfBytes(bytes: Uint8Array, seq: number[], from: number): number {
  for (let i = bytes.indexOf(seq[0]!, from); i !== -1; i = bytes.indexOf(seq[0]!, i + 1)) {
    if (seq.every((byte, n) => bytes[i + n] === byte)) return i;
  }
  return -1;
}

/** Past a CDATA section or comment opening at `i`, whose text may look like a tag: where its
 *  closing `>` is; `i` when neither opens there, -1 when it never closes. */
function skipMarkup(bytes: Uint8Array, i: number): number {
  if (bytes[i + 1] !== 0x21) return i;
  const close = bytes[i + 2] === 0x5b ? CDATA_END : bytes[i + 2] === 0x2d ? COMMENT_END : null;
  if (!close) return i;
  const at = indexOfBytes(bytes, close, i + 3);
  return at === -1 ? -1 : at + 2;
}

/** The next start (or `closing` end) tag with local name `name`, any prefix, from byte `from`:
 *  where it starts, where it ends (after `>`), and whether it closes itself. */
function findTag(bytes: Uint8Array, from: number, name: string, closing: boolean) {
  for (let i = bytes.indexOf(0x3c, from); i !== -1; i = bytes.indexOf(0x3c, i + 1)) {
    const skipped = skipMarkup(bytes, i);
    if (skipped === -1) return null;
    if (skipped !== i) {
      i = skipped;
      continue;
    }
    let j = i + 1;
    if ((bytes[j] === 0x2f) !== closing) continue;
    if (closing) j++;
    // The name runs to whitespace, / or >; a prefix ends at its colon.
    let k = j;
    while (k < bytes.length && bytes[k]! > 0x20 && bytes[k] !== 0x3e && bytes[k] !== 0x2f) {
      if (bytes[k] === 0x3a) j = k + 1;
      k++;
    }
    if (k - j !== name.length) continue;
    let same = true;
    for (let n = 0; n < name.length && same; n++) same = bytes[j + n] === name.charCodeAt(n);
    if (!same) continue;
    const end = bytes.indexOf(0x3e, k);
    if (end === -1) return null;
    return { start: i, end: end + 1, empty: bytes[end - 1] === 0x2f };
  }
  return null;
}

/** Shared strings as cells ask for them: the part is scanned only up to the highest index used,
 *  and each string decoded alone, so a large or stale table costs little. */
function sharedStrings(bytes: Uint8Array | undefined): (index: number) => string {
  // Where each string's XML starts and ends, in flat arrays: a million strings skipped on the way
  // to a high index cost 8 MB, not a million objects.
  let starts = new Uint32Array(1024);
  let ends = new Uint32Array(1024);
  let count = 0;
  const texts = new Map<number, string>();
  let at = bytes ? 0 : -1;
  return (index) => {
    while (bytes && at !== -1 && count <= index) {
      const open = findTag(bytes, at, "si", false);
      const close = open && !open.empty ? findTag(bytes, open.end, "si", true) : null;
      if (!open || (!open.empty && !close)) at = -1;
      else {
        if (count === starts.length) {
          starts = grow(starts);
          ends = grow(ends);
        }
        starts[count] = open.end;
        ends[count] = close ? close.start : open.end;
        count++;
        at = close ? close.end : open.end;
      }
    }
    if (!bytes || !Number.isInteger(index) || index < 0 || index >= count) return "";
    let text = texts.get(index);
    if (text === undefined) {
      // A cell's worth of XML at most: a string cut there ends in an open run, which still reads.
      const start = starts[index]!;
      text = stringText(strFromU8(bytes.subarray(start, Math.min(ends[index]!, start + MAX_CELL_BYTES))));
      texts.set(index, text);
    }
    return text;
  };
}

function grow(array: Uint32Array): Uint32Array<ArrayBuffer> {
  const next = new Uint32Array(array.length * 2);
  next.set(array);
  return next;
}

// A style section past this is skipped: real ones are far smaller, even at Excel's 64,000 formats.
const MAX_STYLE_SECTION_BYTES = 16 * 1024 * 1024;
// A styles part past this is not read: its cells show unstyled.
const MAX_STYLES_BYTES = 3 * MAX_STYLE_SECTION_BYTES;

/** Only the style sections the reader uses (number formats, fonts, cell formats), each found by a
 *  byte scan and decoded alone, so a large styles part costs little. */
function styleSections(bytes: Uint8Array | undefined): Document | null {
  const root = bytes && findTag(bytes, 0, "styleSheet", false);
  if (!bytes || !root || root.empty) return null;
  const open = strFromU8(bytes.subarray(root.start, root.end));
  const parts = ["numFmts", "fonts", "cellXfs"].map((name) => {
    const start = findTag(bytes, root.end, name, false);
    const end = start && !start.empty ? findTag(bytes, start.end, name, true) : null;
    return start && end && end.end - start.start <= MAX_STYLE_SECTION_BYTES
      ? strFromU8(bytes.subarray(start.start, end.end))
      : "";
  });
  const prefix = /^<([\w.-]+:)?/.exec(open)?.[1] ?? "";
  return parseXml(`${open}${parts.join("")}</${prefix}styleSheet>`);
}

export function readXlsx(bytes: Uint8Array): Sheet[] {
  const read = archive(bytes);
  const main = "xl/workbook.xml";
  if ((read.size(main) ?? 0) > MAX_XML_PART_BYTES) throw new Error("File is too large to preview.");
  const head = read([main, relsPath(main)], MAX_XML_PART_BYTES);
  const workbook = xml(head, main);
  if (!workbook) throw new Error("Not a valid XLSX workbook.");
  // Only parts the workbook uses: not custom XML, pivot caches or drawings.
  const rels = relationshipList(head, main);
  const partOf = (type: string, fallback: string) => rels.find((rel) => rel.type.endsWith(`/${type}`))?.path ?? fallback;
  const stringsPath = partOf("sharedStrings", "xl/sharedStrings.xml");
  const stylesPath = partOf("styles", "xl/styles.xml");
  // Each read when a kept cell first needs it: a workbook can carry a large table no cell uses.
  let strings: ((index: number) => string) | undefined;
  const string = (index: number) => (strings ??= sharedStrings(read([stringsPath])[stringsPath]))(index);
  let styles: CellStyle[] | undefined;
  const style = (index: number) =>
    (styles ??= readStyles(styleSections(read([stylesPath], MAX_STYLES_BYTES)[stylesPath])))[index];
  const date1904 = ["1", "true"].includes(first(workbook, "workbookPr")?.getAttribute("date1904") ?? "");
  const paths = new Map(rels.map((rel) => [rel.id, rel.path]));
  const sheets: Sheet[] = [];
  const budget = { cells: MAX_WORKBOOK_CELLS };
  for (const sheet of all(workbook, "sheet")) {
    if (sheet.getAttribute("state") === "hidden" || sheet.getAttribute("state") === "veryHidden") continue;
    if (budget.cells <= 0 || sheets.length >= MAX_SHEETS) {
      // The sheets after this one are left out.
      const last = sheets.at(-1);
      if (last) last.truncated = true;
      break;
    }
    const path = paths.get(relId(sheet, "id") ?? "");
    const part = path ? read([path])[path] : undefined;
    if (!part) continue;
    const { text, cut } = sheetText(part, Math.min(MAX_SHEET_ROWS, budget.cells));
    const parsed = readSheet(text, sheet.getAttribute("name") ?? "Sheet", string, style, date1904, budget);
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
  // A field stops growing at a cell's worth, as an XLSX cell does.
  const append = (char: string) => {
    if (field.length < MAX_CELL_TEXT) field += char;
    else truncated = true;
  };
  for (let i = 0; i < text.length; i++) {
    const char = text[i]!;
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        append('"');
        i++;
      } else if (char === '"') quoted = false;
      else append(char);
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
    } else append(char);
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
  frame?: {
    x: number;
    y: number;
    w: number;
    h: number;
    /** Clockwise, in degrees. */
    rot?: number;
    flipH?: boolean;
    flipV?: boolean;
  };
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
  /** Slides past MAX_SLIDES were left out. */
  truncated?: boolean;
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

/** Image types the package declares (a part's Override, else its extension's Default), for a
 *  picture named without a known extension (image1.bin); only types a browser shows. */
function packageImageTypes(read: Reader): (path: string) => string | undefined {
  const name = "[Content_Types].xml";
  const doc = xml(read([name], MAX_XML_PART_BYTES), name);
  const declared = (tag: string, key: string, trim: RegExp) =>
    new Map(
      (doc ? all(doc, tag) : []).map((el) => [
        (el.getAttribute(key) ?? "").replace(trim, "").toLowerCase(),
        (el.getAttribute("ContentType") ?? "").toLowerCase(),
      ]),
    );
  const defaults = declared("Default", "Extension", /^$/);
  const overrides = declared("Override", "PartName", /^\//);
  const shown = new Set(Object.values(IMAGE_TYPES));
  return (path) => {
    const dot = path.lastIndexOf(".");
    const type = overrides.get(path.toLowerCase()) ?? (dot === -1 ? undefined : defaults.get(path.slice(dot + 1).toLowerCase()));
    return type && shown.has(type) ? type : undefined;
  };
}

// A chart shown as a table, capped so a chart of many long series stays a readable size.
const MAX_CHART_SERIES = 100;
const MAX_CHART_CELLS = 5000;

// Slides read: each is kept as boxes once parsed.
const MAX_SLIDES = 500;
// Characters of text kept across a deck; the slides after are left out.
const MAX_DECK_TEXT = 8 * 1024 * 1024;
// Picture bytes kept across a deck; pictures past it are left out, their slides marked cut.
const MAX_DECK_IMAGE_BYTES = 64 * 1024 * 1024;

function boxText(box: SlideBox): number {
  let n = box.caption?.length ?? 0;
  for (const p of box.paragraphs ?? []) n += p.text.length;
  for (const row of box.table ?? []) for (const cell of row) n += cell.length;
  return n;
}
const HIDDEN_SLIDE = /<(?:[\w.-]+:)?sld\b[^>]*\sshow\s*=\s*["'](?:0|false)["']/;

// Paragraphs, table cells and pictures on one slide, all of which it mounts at once.
const MAX_SLIDE_ITEMS = 10_000;

/** A table's cell text, capped like a chart's and at `limit` cells. */
function readTable(tbl: Element, limit: number): string[][] {
  const table: string[][] = [];
  let cells = 0;
  let cut = false;
  for (const tr of children(tbl, "tr")) {
    const tcs = children(tr, "tc");
    if (tcs.length > MAX_CHART_SERIES) cut = true;
    const row = tcs.slice(0, MAX_CHART_SERIES);
    if (cells + row.length > Math.min(limit, MAX_CHART_CELLS)) {
      cut = true;
      break;
    }
    cells += row.length;
    table.push(row.map((tc) => clip(all(tc, "p").map(paragraphText).join("\n"))));
  }
  if (cut) table.push(["…"]);
  return table;
}

const MAX_DIAGRAM_NODES = 500;

/** A SmartArt diagram's node labels, from its data part, up to `limit`; a last "…" marks more. */
function readDiagram(doc: Document, limit: number): NonNullable<SlideBox["paragraphs"]> {
  const paragraphs: NonNullable<SlideBox["paragraphs"]> = [];
  for (const pt of all(doc, "pt")) {
    // Nodes only: transitions and connections carry no text.
    const type = pt.getAttribute("type");
    if (type && type !== "node") continue;
    const text = clip(all(pt, "p").map(paragraphText).join("\n"));
    if (!text.trim()) continue;
    if (paragraphs.length >= Math.min(limit, MAX_DIAGRAM_NODES)) {
      paragraphs.push({ text: "…" });
      break;
    }
    paragraphs.push({ text, bullet: true });
  }
  return paragraphs;
}

/** A chart's cached data as a table of at most `budget` cells: a header of series names, then one
 *  row a category. A last row of "…" marks data left out. */
function readChart(doc: Document, budget: number): { caption?: string; table: string[][] } | null {
  const serNodes = all(doc, "ser");
  const width = Math.min(serNodes.length, MAX_CHART_SERIES) + 1;
  // Not even the header row fits: the chart is marked left out.
  if (serNodes.length && width > Math.min(budget, MAX_CHART_CELLS)) return { table: [["…"]] };
  // Category rows that fit beside the header row.
  const limit = Math.floor(Math.min(budget, MAX_CHART_CELLS) / width) - 1;
  let cut = serNodes.length > MAX_CHART_SERIES;
  const cache = (node: Element | undefined) => {
    const out: string[] = [];
    for (const pt of node ? all(node, "pt") : []) {
      const idx = Number(pt.getAttribute("idx") ?? out.length);
      if (idx >= limit) cut = true;
      else if (idx >= 0) out[idx] = clip(first(pt, "v")?.textContent ?? "");
    }
    return out;
  };
  const series = serNodes.slice(0, MAX_CHART_SERIES).map((ser) => {
    const tx = children(ser, "tx")[0];
    return {
      name: clip((tx && (cache(tx)[0] ?? first(tx, "v")?.textContent)) ?? ""),
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
  const caption = title ? clip(all(title, "t").map((t) => t.textContent ?? "").join("")) : "";
  return { caption: caption || undefined, table };
}

/** Text kept to a cell's worth, as a spreadsheet cell is. */
const clip = (text: string) => (text.length > MAX_CELL_TEXT ? text.slice(0, MAX_CELL_TEXT) : text);

/** A paragraph's text in order, a manual line break (<a:br/>) kept as a newline. */
function paragraphText(p: Element): string {
  return clip(Array.from(p.children)
    .map((child) =>
      child.localName === "br"
        ? "\n"
        : child.localName === "r" || child.localName === "fld"
          ? all(child, "t").map((t) => t.textContent ?? "").join("")
          : "",
    )
    .join(""));
}

/** An xfrm child's two numbers (off x/y, ext cx/cy); undefined when it has none. */
function point(xfrm: Element | undefined, name: string, a: string, b: string): [number, number] | undefined {
  const node = xfrm && children(xfrm, name)[0];
  return node ? [Number(node.getAttribute(a)) || 0, Number(node.getAttribute(b)) || 0] : undefined;
}

function readFrame(shape: Element, cx: number, cy: number): SlideBox["frame"] {
  const xfrm = first(shape, "xfrm");
  const off = point(xfrm, "off", "x", "y");
  const ext = point(xfrm, "ext", "cx", "cy");
  if (!off || !ext) return undefined;
  let [x, y] = off;
  let [w, h] = ext;
  let rot = degrees(xfrm);
  // Its own flips, each group's flip turning them over again.
  let flipH = isTrue(xfrm?.getAttribute("flipH") ?? null);
  let flipV = isTrue(xfrm?.getAttribute("flipV") ?? null);
  // In a group, a frame is in the group's child space (chOff, chExt): map it out through each
  // group to the slide. A flip mirrors it within the group, turning it the other way; a rotation
  // turns it about the group's centre.
  for (let group = shape.parentElement; group; group = group.parentElement) {
    if (group.localName !== "grpSp") continue;
    const box = children(group, "grpSpPr").flatMap((props) => children(props, "xfrm"))[0];
    const [gx, gy] = point(box, "off", "x", "y") ?? [0, 0];
    const [gw, gh] = point(box, "ext", "cx", "cy") ?? [0, 0];
    const [ox, oy] = point(box, "chOff", "x", "y") ?? [0, 0];
    const [ow, oh] = point(box, "chExt", "cx", "cy") ?? [0, 0];
    const sx = gw && ow ? gw / ow : 1;
    const sy = gh && oh ? gh / oh : 1;
    x = gx + (x - ox) * sx;
    y = gy + (y - oy) * sy;
    w *= sx;
    h *= sy;
    if (isTrue(box?.getAttribute("flipH") ?? null)) {
      x = 2 * gx + gw - x - w;
      rot = -rot;
      flipH = !flipH;
    }
    if (isTrue(box?.getAttribute("flipV") ?? null)) {
      y = 2 * gy + gh - y - h;
      rot = -rot;
      flipV = !flipV;
    }
    const turn = degrees(box);
    if (turn) {
      const angle = (turn * Math.PI) / 180;
      const dx = x + w / 2 - (gx + gw / 2);
      const dy = y + h / 2 - (gy + gh / 2);
      x = gx + gw / 2 + dx * Math.cos(angle) - dy * Math.sin(angle) - w / 2;
      y = gy + gh / 2 + dx * Math.sin(angle) + dy * Math.cos(angle) - h / 2;
      rot += turn;
    }
  }
  rot %= 360;
  return {
    x: x / cx,
    y: y / cy,
    w: w / cx,
    h: h / cy,
    ...(rot ? { rot } : {}),
    ...(flipH ? { flipH } : {}),
    ...(flipV ? { flipV } : {}),
  };
}

/** An xfrm's clockwise rotation in degrees (rot is in 60,000ths). */
function degrees(xfrm: Element | undefined): number {
  return Number(xfrm?.getAttribute("rot")) / 60000 || 0;
}

/** `images: false` reads text only, leaving slide media unpacked. Only the parts visible slides
 *  use are inflated: not notes, comments, masters or unused media. */
export function readPptx(bytes: Uint8Array, { images = true } = {}): Deck {
  const read = archive(bytes);
  const main = "ppt/presentation.xml";
  if ((read.size(main) ?? 0) > MAX_XML_PART_BYTES) throw new Error("File is too large to preview.");
  const head = read([main, relsPath(main)], MAX_XML_PART_BYTES);
  const presentation = xml(head, main);
  if (!presentation) throw new Error("Not a valid PPTX presentation.");
  const size = first(presentation, "sldSz");
  const cx = Number(size?.getAttribute("cx")) || 12192000;
  const cy = Number(size?.getAttribute("cy")) || 6858000;
  const rels = relationships(head, main);
  const slidePaths = all(presentation, "sldId").map((id) => rels.get(relId(id, "id") ?? ""));
  const slides: Slide[] = [];
  // One Blob per picture, however many slides use it.
  const pictures = new Map<string, Blob>();
  let pictureBytes = 0;
  // By extension, else as [Content_Types].xml declares the part (read once, when first needed).
  let declared: ((path: string) => string | undefined) | undefined;
  const pictureType = (path: string) => imageType(path) ?? (declared ??= packageImageTypes(read))(path);
  let truncated = false;
  let textLeft = MAX_DECK_TEXT;
  for (const [index, path] of slidePaths.entries()) {
    if (!path) continue;
    // Past this a slide is not parsed, its DOM too large to build: it shows as cut.
    if ((read.size(path) ?? 0) > MAX_XML_PART_BYTES) {
      if (slides.length === MAX_SLIDES) {
        truncated = true;
        break;
      }
      slides.push({ boxes: [{ paragraphs: [{ text: "…" }] }] });
      continue;
    }
    // One slide at a time, parsed before the next is read: a long deck holds one slide's XML and
    // inflates no more than it shows.
    const part = read([path, relsPath(path)], MAX_XML_PART_BYTES);
    const slideXml = part[path];
    // A hidden slide is left out of the show, so out of the viewer and the model's text too. Read
    // from the root tag, so it is never parsed.
    if (!slideXml || HIDDEN_SLIDE.test(strFromU8(slideXml.subarray(0, 16384)))) continue;
    if (slides.length === MAX_SLIDES) {
      truncated = true;
      break;
    }
    const doc = xml(part, path);
    if (!doc) continue;
    const slideRelList = relationshipList(part, path);
    // A chart, diagram or picture is read as its frame comes, and let go after. One past the part
    // ceiling is left out, and the slide marked cut.
    const partDoc = (partPath: string) => {
      if ((read.size(partPath) ?? 0) > MAX_XML_PART_BYTES) {
        cut = true;
        return null;
      }
      return xml(read([partPath], MAX_XML_PART_BYTES), partPath);
    };
    const slideRels = new Map(slideRelList.map((rel) => [rel.id, rel.path]));
    const boxes: SlideBox[] = [];
    let left = MAX_SLIDE_ITEMS;
    let cut = false;
    // Each adder returns true once the budget is spent, which ends the slide.
    const addShape = (shape: Element): boolean => {
      const body = first(shape, "txBody");
      if (!body) return false;
      if (left <= 0) return (cut = true);
      const paragraphs: NonNullable<SlideBox["paragraphs"]> = [];
      // Read only as far as the budget: a shape can hold far more paragraphs than are kept.
      const list = body.getElementsByTagNameNS("*", "p");
      for (let i = 0, p = list.item(0); p; p = list.item(++i)) {
        const text = paragraphText(p);
        if (!text.trim()) continue;
        if (paragraphs.length === left) {
          cut = true;
          break;
        }
        const runs = all(p, "r");
        const rPr = runs[0] && first(runs[0], "rPr");
        const pPr = first(p, "pPr");
        const sz = Number(rPr?.getAttribute("sz"));
        paragraphs.push({
          text,
          size: sz ? sz / 100 : undefined,
          bold: isTrue(rPr?.getAttribute("b") ?? null),
          align: pPr?.getAttribute("algn") ?? undefined,
          bullet: Boolean(pPr && (first(pPr, "buChar") || first(pPr, "buAutoNum"))),
        });
      }
      if (!paragraphs.length) return false;
      left -= paragraphs.length;
      boxes.push({
        frame: readFrame(shape, cx, cy),
        placeholder: first(shape, "ph")?.getAttribute("type") ?? (first(shape, "ph") ? "body" : undefined),
        paragraphs,
      });
      return false;
    };
    // Tables, charts and SmartArt sit in a graphicFrame. A chart shows its cached data.
    const addFrame = (frame: Element): boolean => {
      if (left <= 0) return (cut = true);
      const place = () => readFrame(frame, cx, cy) ?? { x: 0.05, y: 0.25, w: 0.9, h: 0.65 };
      const chartRef = first(frame, "chart");
      const chartPath = chartRef && slideRels.get(relId(chartRef, "id") ?? "");
      const chartDoc = chartPath ? partDoc(chartPath) : null;
      const chart = chartDoc && readChart(chartDoc, left);
      if (chart) {
        boxes.push({ frame: place(), ...chart });
        left -= chart.table.reduce((n, row) => n + row.length, 0);
      }
      const diagramRef = first(frame, "relIds");
      const diagramPath = diagramRef && slideRels.get(relId(diagramRef, "dm") ?? "");
      const diagramDoc = diagramPath ? partDoc(diagramPath) : null;
      const diagram = diagramDoc && left > 0 ? readDiagram(diagramDoc, left) : [];
      if (diagram.length) {
        boxes.push({ frame: place(), paragraphs: diagram });
        left -= diagram.length;
      }
      const tbl = first(frame, "tbl");
      if (!tbl || left <= 0) return false;
      const table = readTable(tbl, left);
      if (!table.some((row) => row.some((cell) => cell.trim()))) return false;
      boxes.push({ frame: place(), table });
      left -= table.reduce((n, row) => n + row.length, 0);
      return false;
    };
    const addPicture = (pic: Element): boolean => {
      const blip = first(pic, "blip");
      const target = blip && slideRels.get(relId(blip, "embed") ?? "");
      const type = target && pictureType(target);
      if (!target || !type) return false;
      if (left <= 0) return (cut = true);
      let image = pictures.get(target);
      if (!image) {
        // Pictures stay with the deck, so their bytes are bounded across it.
        if (pictureBytes + (read.size(target) ?? 0) > MAX_DECK_IMAGE_BYTES) {
          cut = true;
          return false;
        }
        const data = read([target])[target];
        if (!data) return false;
        pictureBytes += data.length;
        image = new Blob([data as Uint8Array<ArrayBuffer>], { type });
        pictures.set(target, image);
      }
      left--;
      boxes.push({ frame: readFrame(pic, cx, cy), image });
      return false;
    };
    // In document order, as PowerPoint stacks them: a later shape draws over an earlier one.
    const walk = (parent: Element): boolean => {
      for (let node = parent.firstElementChild; node; node = node.nextElementSibling) {
        const name = node.localName;
        const branch =
          name === "AlternateContent" ? (children(node, "Fallback")[0] ?? children(node, "Choice")[0]) : undefined;
        const stop =
          name === "grpSp"
            ? walk(node)
            : branch
              ? walk(branch)
              : name === "sp"
                ? addShape(node)
                : name === "graphicFrame"
                  ? addFrame(node)
                  : name === "pic" && images
                    ? addPicture(node)
                    : false;
        if (stop) return true;
      }
      return false;
    };
    const tree = first(doc, "spTree");
    if (tree) walk(tree);
    // What was left out, marked as a table's cut is.
    if (cut) boxes.push({ paragraphs: [{ text: "…" }] });
    slides.push({ boxes });
    // Paragraphs are counted, but one can be long: the deck's text is bounded by size too.
    textLeft -= boxes.reduce((n, box) => n + boxText(box), 0);
    if (textLeft <= 0) {
      truncated = index < slidePaths.length - 1;
      break;
    }
  }
  return { aspect: cy / cx, widthPt: cx / 12700, slides, truncated };
}
