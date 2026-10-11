// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Inflate, type Unzipped, inflateSync, strFromU8, unzipSync } from "fflate";


const MAX_UNPACKED_BYTES = 200 * 1024 * 1024;
const MAX_XML_PART_BYTES = 10 * 1024 * 1024;
const MAX_ZIP_ENTRIES = 100_000;
export const MAX_SHEET_ROWS = 5000;
export const MAX_SHEET_COLUMNS = 200;
const MAX_WORKBOOK_CELLS = MAX_SHEET_ROWS * (MAX_SHEET_COLUMNS + 1);
const MAX_SHEETS = 100;

export interface SheetLimits {
  sheets: number;
  rows: number;
  columns: number;
  /** Sheets past the limit listed by name only, unparsed, up to this many. */
  extraNames?: number;
}

const FULL_SHEET_LIMITS: SheetLimits = { sheets: MAX_SHEETS, rows: MAX_SHEET_ROWS, columns: MAX_SHEET_COLUMNS };

export interface SheetCell {
  text: string;
  numeric?: boolean;
  bold?: boolean;
  italic?: boolean;
}

export interface Sheet {
  name: string;
  rows: (SheetCell | undefined)[][];
  widths: (number | undefined)[];
  truncated: boolean;
  hidden?: { rows: Set<number>; columns: Set<number> };
}

interface ZipEntry {
  offset: number;
  size: number;
  originalSize: number;
  method: number;
}

function zipIndex(bytes: Uint8Array, view: DataView): Map<string, ZipEntry> | null {
  let end = bytes.length - 22;
  const stop = Math.max(0, end - 0xffff);
  while (end >= stop && view.getUint32(end, true) !== 0x06054b50) end--;
  if (end < stop) return null;
  let count = view.getUint16(end + 10, true);
  let at = view.getUint32(end + 16, true);
  const u64 = (offset: number) => (offset + 8 <= bytes.length ? Number(view.getBigUint64(offset, true)) : Number.NaN);
  if (count === 0xffff || at === 0xffffffff) {
    // ZIP64: the real count and directory offset are in the record its locator points at.
    const locator = end - 20;
    if (locator < 0 || view.getUint32(locator, true) !== 0x07064b50) return null;
    const record = u64(locator + 8);
    if (!(record + 56 <= bytes.length) || view.getUint32(record, true) !== 0x06064b50) return null;
    count = u64(record + 32);
    at = u64(record + 48);
    if (!Number.isSafeInteger(count) || !Number.isSafeInteger(at)) return null;
  }
  if (count > MAX_ZIP_ENTRIES) throw new Error("File is too large to preview.");
  const utf8 = new TextDecoder();
  const index = new Map<string, ZipEntry>();
  for (let i = 0; i < count; i++) {
    if (at + 46 > bytes.length || view.getUint32(at, true) !== 0x02014b50) return null;
    let size = view.getUint32(at + 20, true);
    let originalSize = view.getUint32(at + 24, true);
    let offset = view.getUint32(at + 42, true);
    const nameEnd = at + 46 + view.getUint16(at + 28, true);
    const extraEnd = nameEnd + view.getUint16(at + 30, true);
    if (size === 0xffffffff || originalSize === 0xffffffff || offset === 0xffffffff) {
      // The ZIP64 extra field (0x0001) holds, in order, the 64-bit values that did not fit.
      let field = -1;
      for (let extra = nameEnd; extra + 4 <= extraEnd; extra += 4 + view.getUint16(extra + 2, true)) {
        if (view.getUint16(extra, true) === 1) {
          field = extra + 4;
          break;
        }
      }
      if (field === -1) return null;
      const next = () => {
        const value = field + 8 <= extraEnd ? u64(field) : Number.NaN;
        field += 8;
        return value;
      };
      if (originalSize === 0xffffffff) originalSize = next();
      if (size === 0xffffffff) size = next();
      if (offset === 0xffffffff) offset = next();
      if (![size, originalSize, offset].every(Number.isSafeInteger)) return null;
    }
    const raw = bytes.subarray(at + 46, nameEnd);
    // Bit 11 marks a UTF-8 name; otherwise one byte a character, as unzipSync reads it.
    const name = view.getUint16(at + 8, true) & 0x800 ? utf8.decode(raw) : String.fromCharCode(...raw);
    index.set(name, { offset, size, originalSize, method: view.getUint16(at + 10, true) });
    at = extraEnd + view.getUint16(at + 32, true);
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

type Reader = ((names: Iterable<string>, limit?: number) => Unzipped) & {
  size: (name: string) => number | undefined;
  head: (name: string, max: number) => Uint8Array | undefined;
  open: (name: string, max: number) => Growing | undefined;
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
    let sizes: Map<string, number> | undefined;
    const size = (name: string) => {
      if (!sizes) {
        const found = new Map<string, number>();
        unzipSync(bytes, { filter: (entry) => (found.set(entry.name, entry.originalSize), false) });
        sizes = found;
      }
      return sizes.get(name);
    };
    const head = (name: string, max: number) => {
      if ((size(name) ?? 0) > max) throw new Error("File is too large to preview.");
      return read([name])[name];
    };
    const open = (name: string, max: number): Growing | undefined => {
      const data = head(name, max);
      return data && { data, done: true, grow() {} };
    };
    return Object.assign(read, { size, head, open });
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
  const open = (name: string, max: number) => {
    const entry = index.get(name);
    return entry && growing(bytes, view, entry, max, charge);
  };
  const head = (name: string, max: number) => {
    const entry = index.get(name);
    if (!entry || entry.originalSize <= max) return read([name])[name];
    const part = growing(bytes, view, entry, max, charge);
    while (!part.done) part.grow();
    return part.data;
  };
  return Object.assign(read, { size: (name: string) => index.get(name)?.originalSize, head, open });
}

export interface Growing {
  data: Uint8Array;
  done: boolean;
  cut?: boolean;
  grow(): void;
}

function growing(bytes: Uint8Array, view: DataView, entry: ZipEntry, max: number, charge: (size: number) => void): Growing {
  const at = entry.offset;
  if (view.getUint32(at, true) !== 0x04034b50) throw new Error("Not a valid ZIP archive.");
  const start = at + 30 + view.getUint16(at + 26, true) + view.getUint16(at + 28, true);
  const data = bytes.subarray(start, start + entry.size);
  if (entry.method === 0) {
    const whole = data.slice(0, max);
    charge(whole.length);
    return { data: whole, done: true, cut: data.length > max, grow() {} };
  }
  if (entry.method !== 8) throw new Error(`Unsupported ZIP compression method ${entry.method}.`);
  let buffer = new Uint8Array(Math.min(max, 1 << 20));
  let length = 0;
  let from = 0;
  const inflater = new Inflate((chunk) => {
    const room = Math.min(chunk.length, max - length);
    if (room < chunk.length) part.cut = true;
    if (length + room > buffer.length) {
      const next = new Uint8Array(Math.min(max, Math.max(buffer.length * 2, length + room)));
      next.set(buffer.subarray(0, length));
      buffer = next;
    }
    buffer.set(chunk.subarray(0, room), length);
    length += room;
  });
  const part: Growing = {
    data: buffer.subarray(0, 0),
    done: false,
    grow() {
      const before = length;
      while (!part.done && length < before + (1 << 20)) {
        inflater.push(data.subarray(from, from + 16384), from + 16384 >= data.length);
        from += 16384;
        if (from >= data.length || length >= max) part.done = true;
      }
      charge(length - before);
      part.data = buffer.subarray(0, length);
    },
  };
  return part;
}

function xml(files: Unzipped, path: string): Document | null {
  const bytes = files[path];
  return bytes ? parseXml(strFromU8(bytes)) : null;
}

function parseXml(text: string): Document | null {
  const doc = new DOMParser().parseFromString(text, "application/xml");
  return doc.getElementsByTagName("parsererror").length ? null : doc;
}

function all(node: Document | Element, name: string): Element[] {
  return Array.from(node.getElementsByTagNameNS("*", name));
}

function first(node: Document | Element, name: string): Element | undefined {
  return node.getElementsByTagNameNS("*", name)[0];
}

function children(node: Element, name: string): Element[] {
  return Array.from(node.children).filter((child) => child.localName === name);
}

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

function relationships(files: Unzipped, part: string): Map<string, string> {
  return new Map(relationshipList(files, part).map((rel) => [rel.id, rel.path]));
}

function mainPart(read: Reader, fallback: string): string {
  const root = read(["_rels/.rels"], MAX_XML_PART_BYTES);
  const path = relationshipList(root, "").find((rel) => rel.type.endsWith("/officeDocument"))?.path;
  return path && read.size(path) !== undefined ? path : fallback;
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

function relId(node: Element, name: string): string | null {
  for (const attr of Array.from(node.attributes)) {
    if (attr.localName === name && attr.namespaceURI?.endsWith("/relationships")) return attr.value;
  }
  return null;
}


const BUILTIN_FORMATS: Record<number, string> = {
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

function memo<T>(compute: (code: string) => T): (code: string) => T {
  const cache = new Map<string, T>();
  return (code) => {
    let value = cache.get(code);
    if (value === undefined) {
      if (cache.size >= 8192) cache.clear();
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

/** `m` is minutes after an hour or before a second, otherwise the month. */
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

const fractionShape = memo((format) =>
  /(?:([0#?]+)\s+)?[0#?]+\s*\/\s*([1-9]\d*|[0#?]+)/.exec(format.replace(/"[^"]*"|\\./g, (text) => "\u0001".repeat(text.length))),
);

const unquote = (format: string) => format.replace(/"([^"]*)"/g, "$1").replace(/\\(.)/g, "$1");

function closestFraction(x: number, maxDen: number): [number, number] {
  let [p0, q0, p1, q1] = [0, 1, 1, 0];
  let v = x;
  for (let i = 0; i < 64; i++) {
    const a = Math.floor(v);
    const q2 = q0 + a * q1;
    if (q2 > maxDen) break;
    [p0, q0, p1, q1] = [p1, q1, p0 + a * p1, q2];
    if (v - a < 1e-12) break;
    v = 1 / (v - a);
  }
  const k = Math.floor((maxDen - q0) / q1);
  const [ps, qs] = [p0 + k * p1, q0 + k * q1];
  return Math.abs(x - ps / qs) < Math.abs(x - p1 / q1) - 1e-12 ? [ps, qs] : [p1, q1];
}

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
    [numerator, denominator] = closestFraction(rest, 10 ** denominatorCode.length - 1);
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

function formatElapsed(value: number, format: string): string {
  const tokens: string[] = format.match(/"[^"]*"|\\.|\[[hms]+\]|h+|m+|s+|\.0+|[\s\S]/gi) ?? [];
  const digits = tokens.find((token) => /^\.0+$/.test(token))?.length ?? 1;
  const scale = 10 ** (digits - 1);
  const ticks = Math.round(Math.abs(value) * 86400 * scale);
  const total = Math.floor(ticks / scale);
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
  return String(Number.isInteger(value) ? value : Number(value.toPrecision(15)));
}

const isPlaceholder = (token: string) => token === "0" || token === "#" || token === "?";
const emptyPlaceholder = (token: string) => (token === "0" ? "0" : token === "?" ? " " : "");

function literalText(token: string): string {
  if (token.startsWith('"')) return token.slice(1, -1);
  if (token.startsWith("\\")) return token.slice(1);
  return token === "," ? "" : token;
}

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
    grouped: whole.some((token, i) => token === "," && i > first && i < last),
    zeros: whole.filter((token) => token === "0").length,
    before: whole.slice(0, first).map(literalText).join(""),
    after: whole.slice(last + 1).map(literalText).join(""),
  };
});

function placeDigits(number: number, format: string): string {
  const { dot, whole, part, first, places, grouped, zeros, before, after } = digitLayout(format);
  const shown = number ? 14 - Math.floor(Math.log10(Math.abs(number))) : 100;
  const [intDigits = "", fracDigits = ""] = number
    .toFixed(Math.max(0, Math.min(places.length, shown, 100)))
    .split(".");
  if (/e/i.test(intDigits)) return generalText(number);
  const significant = intDigits === "0" ? "" : intDigits;
  let integer: string;
  // No whole-number placeholders (".00"): Excel still shows the integer, just left of the point.
  if (first === -1) {
    integer = `${whole.map(literalText).join("")}${significant}`;
  } else if (grouped) {
    integer = `${before}${significant.padStart(zeros, "0").replace(/\B(?=(\d{3})+$)/g, ",")}${after}`;
  } else {
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
  const digits = fracDigits.padEnd(places.length, "0").split("");
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

function pickSection(value: number, sections: string[]): { index: number; sign: string } {
  const first = meets(value, sections[0]);
  const second = meets(value, sections[1]);
  if (first === null && second === null) {
    const index = value < 0 && sections.length > 1 ? 1 : value === 0 && sections.length > 2 ? 2 : 0;
    return { index, sign: value < 0 && index === 0 ? "-" : "" };
  }
  const sign = value < 0 ? "-" : "";
  if (first) return { index: 0, sign };
  if (second || (second === null && sections.length > 1)) return { index: 1, sign };
  return { index: Math.min(2, sections.length - 1), sign };
}

/** `date1904`: dates count from 1904, 1,462 days after the 1900 system. */
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

const readSection = memo((section) => {
  const tagged = (section.match(/"[^"]*"|\\[\s\S]|\[[^\]]*\]|[_*][\s\S]?|[\s\S]/g) ?? [])
    .map((token) => {
      if (token[0] === "_" || token[0] === "*") return "";
      if (token[0] !== "[" || /^\[[hms]+\]$/i.test(token)) return token;
      const symbol = /^\[\$([^\]-]*)/.exec(token)?.[1];
      return symbol ? `"${symbol}"` : "";
    })
    .join("");
  const code = tagged.replace(/"([^"]*)"/g, "$1").replace(/\\(.)/g, "$1");
  const bare = tagged.replace(/"[^"]*"|\\./g, "");
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
  const scale = 1000 ** (bare.match(SCALE_COMMAS)?.[2]?.length ?? 0);
  return { tagged, code, kind, percents: bare.split("%").length - 1, scale };
});

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

function formatScientific(value: number, format: string): string {
  const at = exponentAt(format);
  const mantissaFormat = format.slice(0, at);
  const exponentFormat = format.slice(at + 2);
  const tokens: string[] = mantissaFormat.match(/"[^"]*"|\\.|[0#?.]|[^"\\0#?.]+/g) ?? [];
  const dot = tokens.indexOf(".");
  const places = Math.min(dot === -1 ? 0 : tokens.slice(dot + 1).filter(isPlaceholder).length, 100);
  const step = Math.max(1, (dot === -1 ? tokens : tokens.slice(0, dot)).filter(isPlaceholder).length);
  const [normalized = "0", exponentText = "0"] = value.toExponential(places).split("e");
  let mantissa = normalized;
  let exponent = Number(exponentText);
  if (step > 1 && value !== 0) {
    exponent = Math.floor(Math.log10(value) / step) * step;
    mantissa = (value / 10 ** exponent).toFixed(places);
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
  if (kind === "literal") return `${sign}${code.replace(/general/i, generalText(Math.abs(value)))}`.trim();
  const number = (Math.abs(value) * 100 ** percents) / scale;
  const unscaled = (format: string) => (scale === 1 ? format : format.replace(SCALE_COMMAS, "$1"));
  if (kind === "scientific") return `${sign}${formatScientific(number, unscaled(tagged)).trim()}`;
  const fraction = formatFraction(number, unscaled(tagged));
  if (fraction !== null) return `${sign}${fraction}`;
  return `${sign}${placeDigits(number, tagged).trim()}`;
}

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
    const code = char.charCodeAt(0) & ~32;
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

function localeFormat(id: number): string | undefined {
  if (id === 32) return "h:mm";
  if (id === 33) return "h:mm:ss";
  return (id >= 27 && id <= 36) || (id >= 50 && id <= 58) ? "m/d/yyyy" : undefined;
}

const isOn = (flag: Element) => !["0", "false", "off"].includes(flag.getAttribute("val") ?? "");

const MAX_FORMAT_CODE = 1024;
const MAX_NUM_FMTS = 1000;

function readStyles(doc: Document | null): { styles: CellStyle[]; cut: boolean } {
  if (!doc) return { styles: [], cut: false };
  const formats = new Map<number, string>();
  let cut = false;
  for (const [index, fmt] of all(doc, "numFmt").entries()) {
    const code = fmt.getAttribute("formatCode") ?? "";
    // Past Excel's own limits a format is read as General: the formatters cache by code, across files.
    const kept = index < MAX_NUM_FMTS && code.length <= MAX_FORMAT_CODE;
    cut ||= !kept;
    formats.set(Number(fmt.getAttribute("numFmtId")), kept ? code : "General");
  }
  const fontsNode = first(doc, "fonts");
  const fonts = fontsNode
    ? children(fontsNode, "font").map((font) => ({
        bold: children(font, "b").some(isOn),
        italic: children(font, "i").some(isOn),
      }))
    : [];
  const xfs = first(doc, "cellXfs");
  const styles = (xfs ? children(xfs, "xf") : []).map((xf) => {
    const id = Number(xf.getAttribute("numFmtId") ?? 0);
    const font = fonts[Number(xf.getAttribute("fontId") ?? 0)];
    return { format: formats.get(id) ?? BUILTIN_FORMATS[id] ?? localeFormat(id), ...font };
  });
  return { styles, cut };
}

const REFERENCE =
  /(^|[^A-Za-z0-9_.$])(?:(\$?)([A-Za-z]{1,3})(\$?)(\d+)(?![\d(A-Za-z_!])|(\$?)([A-Za-z]{1,3}):(\$?)([A-Za-z]{1,3})(?![\w(!.])|(\$?)(\d+):(\$?)(\d+)(?![\d.]))/g;

function shiftFormula(formula: string, rows: number, columns: number): string {
  const column = (abs: string, name: string) =>
    abs + (abs ? name.toUpperCase() : columnName(columnIndex(name.toUpperCase()) + columns));
  const row = (abs: string, n: string) => abs + (abs ? n : String(Number(n) + rows));
  return formula
    .split(/("(?:[^"]|"")*"|'(?:[^']|'')*'|\[(?:[^[\]']|'.|\[(?:[^[\]']|'.)*\])*\])/)
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


const NAMED_ENTITIES: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };
const ENTITY = /&(?:#x([0-9a-f]+)|#(\d+)|(amp|lt|gt|quot|apos));|<!\[CDATA\[([\s\S]*?)\]\]>/gi;

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

function attribute(attributes: string, name: string): string | null {
  let pattern = attributePatterns.get(name);
  if (!pattern) {
    pattern = new RegExp(`(?:^|\\s)${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`);
    attributePatterns.set(name, pattern);
  }
  const match = pattern.exec(attributes);
  return match ? decodeXml(match[1] ?? match[2] ?? "") : null;
}

const openTag = (name: string) => new RegExp(`<(?:[\\w.-]+:)?${name}(?=[\\s/>])([^>]*?)(/)?>`, "g");
const closeTag = (name: string) => new RegExp(`</(?:[\\w.-]+:)?${name}\\s*>`, "g");

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

function flattenMarkup(text: string): string {
  if (!text.includes("<!")) return text;
  return text.replace(MARKUP, (_, cdata?: string) =>
    cdata === undefined ? "" : cdata.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"),
  );
}

function stringText(xmlText: string): string {
  let out = "";
  for (const match of flattenMarkup(xmlText).replace(PHONETIC, "").matchAll(TEXT_RUN)) {
    out += decodeXml(match[1]!.slice(0, MAX_CELL_BYTES));
    if (out.length >= MAX_CELL_TEXT) return out.slice(0, MAX_CELL_TEXT);
  }
  return out;
}

const isTrue = (value: string | null) => value === "1" || value === "true";

const MAX_CELL_TEXT = 32_767;
const MAX_CELL_BYTES = MAX_CELL_TEXT * 10;

function readSheet(
  text: string,
  name: string,
  string: (index: number) => string,
  style: (index: number) => CellStyle | undefined,
  date1904: boolean,
  budget: { cells: number },
  limits: SheetLimits,
): Sheet {
  const rows: (SheetCell | undefined)[][] = [];
  const widths: (number | undefined)[] = [];
  const hidden = { rows: new Set<number>(), columns: new Set<number>() };
  text = flattenMarkup(text);
  const dataAt = text.search(SHEET_DATA);
  const head = dataAt === -1 ? text : text.slice(0, dataAt);
  for (const [attrs] of elements(head, COL_OPEN, COL_CLOSE)) {
    const min = Number(attribute(attrs, "min") ?? 1);
    const max = Math.min(Number(attribute(attrs, "max") ?? min), limits.columns);
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
    if (r >= limits.rows || budget.cells <= 0) {
      truncated = true;
      break;
    }
    budget.cells--;
    const rowHidden = isTrue(attribute(rowAttrs, "hidden"));
    if (rowHidden) hidden.rows.add(r);
    if (rowHidden && !rowBody.includes("shared")) continue;
    const cells: (SheetCell | undefined)[] = [];
    let nextColumn = 0;
    for (const [attrs, body] of elements(rowBody, CELL_OPEN, CELL_CLOSE)) {
      const ref = attribute(attrs, "r");
      const col = ref ? columnIndex(ref) : nextColumn;
      nextColumn = col + 1;
      if (col >= limits.columns) {
        truncated ||= !rowHidden;
        continue;
      }
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

const SHEET_CUT_BYTES = 1024 * 1024;
const MAX_SHEET_XML_BYTES = 64 * 1024 * 1024;

const SHEET_DATA_CLOSE = Array.from("sheetData>", (char) => char.charCodeAt(0));

function rowsEnd(bytes: Uint8Array, rows: number): { end: number; capped: boolean; closed?: number } {
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
    } else if (SHEET_DATA_CLOSE.every((byte, n) => bytes[name + n] === byte)) {
      return { end: -1, capped: false, closed: name + SHEET_DATA_CLOSE.length };
    }
  }
  return { end: -1, capped: false };
}

function sheetText(bytes: Uint8Array, rows: number, whole = true): { text: string; cut: boolean } {
  if (bytes.length <= SHEET_CUT_BYTES) return { text: strFromU8(bytes), cut: false };
  const { end, capped, closed } = rowsEnd(bytes, rows);
  if (closed !== undefined) return { text: strFromU8(bytes.subarray(0, closed)), cut: false };
  if (end === -1) {
    return bytes.length > MAX_SHEET_XML_BYTES || capped
      ? { text: strFromU8(bytes.subarray(0, MAX_SHEET_XML_BYTES)), cut: true }
      : { text: strFromU8(bytes), cut: false };
  }
  const next = rowsEnd(bytes.subarray(end), 1);
  const more = capped || next.end !== -1 || (!whole && next.closed === undefined);
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

function skipMarkup(bytes: Uint8Array, i: number): number {
  if (bytes[i + 1] !== 0x21) return i;
  const close = bytes[i + 2] === 0x5b ? CDATA_END : bytes[i + 2] === 0x2d ? COMMENT_END : null;
  if (!close) return i;
  const at = indexOfBytes(bytes, close, i + 3);
  return at === -1 ? -1 : at + 2;
}

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

const MAX_SHARED_STRINGS = 2_000_000;

function sharedStrings(part: Growing | undefined): (index: number) => string | undefined {
  let starts = new Uint32Array(1024);
  let ends = new Uint32Array(1024);
  let count = 0;
  const texts = new Map<number, string>();
  let at = part ? 0 : -1;
  return (index) => {
    while (part && at !== -1 && count <= index && count < MAX_SHARED_STRINGS) {
      const open = findTag(part.data, at, "si", false);
      const close = open && !open.empty ? findTag(part.data, open.end, "si", true) : null;
      if (!open || (!open.empty && !close)) {
        if (part.done) at = -1;
        else part.grow();
        continue;
      }
      if (count === starts.length) {
        starts = grow(starts);
        ends = grow(ends);
      }
      starts[count] = open.end;
      ends[count] = close ? close.start : open.end;
      count++;
      at = close ? close.end : open.end;
    }
    if ((part?.cut || count >= MAX_SHARED_STRINGS) && index >= count) return undefined;
    if (!part || !Number.isInteger(index) || index < 0 || index >= count) return "";
    let text = texts.get(index);
    if (text === undefined) {
      const start = starts[index]!;
      text = stringText(strFromU8(part.data.subarray(start, Math.min(ends[index]!, start + MAX_CELL_BYTES))));
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

const MAX_STYLE_SECTION_BYTES = 16 * 1024 * 1024;
const MAX_STYLE_SECTION_TAGS = 200_000;
const MAX_STYLES_BYTES = 3 * MAX_STYLE_SECTION_BYTES;

function styleSections(bytes: Uint8Array | undefined): { doc: Document | null; cut: boolean } {
  const root = bytes && findTag(bytes, 0, "styleSheet", false);
  if (!bytes || !root || root.empty) return { doc: null, cut: false };
  const open = strFromU8(bytes.subarray(root.start, root.end));
  let cut = false;
  const parts = ["numFmts", "fonts", "cellXfs"].map((name) => {
    const start = findTag(bytes, root.end, name, false);
    const end = start && !start.empty ? findTag(bytes, start.end, name, true) : null;
    if (!start || !end) return "";
    if (end.end - start.start > MAX_STYLE_SECTION_BYTES) return ((cut = true), "");
    // Counted before it is parsed: tiny elements would build a DOM far larger than the bytes.
    let tags = 0;
    for (let at = start.start; at < end.end && tags <= MAX_STYLE_SECTION_TAGS; at++) {
      if (bytes[at] === 0x3c) tags++;
    }
    if (tags > MAX_STYLE_SECTION_TAGS) return ((cut = true), "");
    return strFromU8(bytes.subarray(start.start, end.end));
  });
  const prefix = /^<([\w.-]+:)?/.exec(open)?.[1] ?? "";
  return { doc: parseXml(`${open}${parts.join("")}</${prefix}styleSheet>`), cut };
}

export function readXlsx(bytes: Uint8Array, limits: SheetLimits = FULL_SHEET_LIMITS): Sheet[] {
  const read = archive(bytes);
  const main = mainPart(read, "xl/workbook.xml");
  if ((read.size(main) ?? 0) > MAX_XML_PART_BYTES) throw new Error("File is too large to preview.");
  const head = read([main, relsPath(main)], MAX_XML_PART_BYTES);
  const workbook = xml(head, main);
  if (!workbook) throw new Error("Not a valid XLSX workbook.");
  const rels = relationshipList(head, main);
  const partOf = (type: string, fallback: string) => rels.find((rel) => rel.type.endsWith(`/${type}`))?.path ?? fallback;
  const stringsPath = partOf("sharedStrings", "xl/sharedStrings.xml");
  const stylesPath = partOf("styles", "xl/styles.xml");
  let strings: ((index: number) => string | undefined) | undefined;
  let missed = false;
  const string = (index: number) => {
    const text = (strings ??= sharedStrings(read.open(stringsPath, MAX_SHEET_XML_BYTES)))(index);
    missed ||= text === undefined;
    return text ?? "";
  };
  let styles: CellStyle[] | undefined;
  let stylesCut = false;
  const style = (index: number) => {
    if (!styles) {
      const sections = styleSections(read([stylesPath], MAX_STYLES_BYTES)[stylesPath]);
      const table = readStyles(sections.doc);
      stylesCut = sections.cut || table.cut || (read.size(stylesPath) ?? 0) > MAX_STYLES_BYTES;
      styles = table.styles;
    }
    missed ||= stylesCut;
    return styles[index];
  };
  const date1904 = ["1", "true"].includes(first(workbook, "workbookPr")?.getAttribute("date1904") ?? "");
  const paths = new Map(rels.map((rel) => [rel.id, rel.path]));
  const sheets: Sheet[] = [];
  const budget = { cells: MAX_WORKBOOK_CELLS };
  for (const sheet of all(workbook, "sheet")) {
    if (sheet.getAttribute("state") === "hidden" || sheet.getAttribute("state") === "veryHidden") continue;
    if (limits.extraNames !== undefined && sheets.length >= limits.sheets) {
      if (sheets.length >= limits.sheets + limits.extraNames) break;
      sheets.push({ name: sheet.getAttribute("name") ?? "Sheet", rows: [], widths: [], truncated: false });
      continue;
    }
    if (budget.cells <= 0 || sheets.length >= limits.sheets) {
      const last = sheets.at(-1);
      if (last) last.truncated = true;
      break;
    }
    const path = paths.get(relId(sheet, "id") ?? "");
    const part = path ? read.head(path, MAX_SHEET_XML_BYTES + 1) : undefined;
    if (!part) continue;
    const whole = part.length <= MAX_SHEET_XML_BYTES;
    const { text, cut } = sheetText(part, Math.min(limits.rows, budget.cells), whole);
    missed = false;
    const parsed = readSheet(text, sheet.getAttribute("name") ?? "Sheet", string, style, date1904, budget, limits);
    if (cut || missed) parsed.truncated = true;
    sheets.push(parsed);
  }
  return sheets;
}

export function readDelimited(
  text: string,
  delimiter: string,
  name: string,
  limits: SheetLimits = FULL_SHEET_LIMITS,
): Sheet {
  const rows: (SheetCell | undefined)[][] = [];
  let row: (SheetCell | undefined)[] = [];
  let field = "";
  let quoted = false;
  let truncated = false;
  const push = () => {
    if (row.length < limits.columns) {
      const numeric = field.trim() !== "" && Number.isFinite(Number(field.replace(/[$,%]/g, "")));
      row.push(field === "" ? undefined : { text: field, numeric });
    } else truncated = true;
    field = "";
  };
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
      if (rows.length >= limits.rows) {
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


export interface SlideBox {
  frame?: {
    x: number;
    y: number;
    w: number;
    h: number;
    rot?: number;
    flipH?: boolean;
    flipV?: boolean;
  };
  placeholder?: string;
  paragraphs?: { text: string; size?: number; bold?: boolean; align?: string; bullet?: boolean }[];
  image?: Blob;
  crop?: { l: number; t: number; r: number; b: number };
  table?: string[][];
  caption?: string;
}

export interface Slide {
  boxes: SlideBox[];
}

export interface Deck {
  aspect: number;
  widthPt: number;
  slides: Slide[];
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

const MAX_CHART_SERIES = 100;
const MAX_CHART_CELLS = 5000;

const MAX_SLIDES = 500;
const MAX_DECK_TEXT = 8 * 1024 * 1024;
const MAX_DECK_IMAGE_BYTES = 64 * 1024 * 1024;
const MAX_PICTURE_PIXELS = 64 * 1024 * 1024;
const MAX_SLIDE_PIXELS = 128 * 1024 * 1024;

function imagePixels(b: Uint8Array): number | undefined {
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const ascii = (at: number, text: string) => [...text].every((c, i) => b[at + i] === c.charCodeAt(0));
  if (b.length >= 24 && ascii(1, "PNG")) return view.getUint32(16) * view.getUint32(20);
  if (b.length >= 10 && ascii(0, "GIF8")) return view.getUint16(6, true) * view.getUint16(8, true);
  if (b.length >= 26 && ascii(0, "BM")) {
    if (view.getUint32(14, true) === 12) return view.getUint16(18, true) * view.getUint16(20, true);
    return Math.abs(view.getInt32(18, true)) * Math.abs(view.getInt32(22, true));
  }
  if (b.length >= 30 && ascii(0, "RIFF") && ascii(8, "WEBP")) {
    if (ascii(12, "VP8 ")) return (view.getUint16(26, true) & 0x3fff) * (view.getUint16(28, true) & 0x3fff);
    if (ascii(12, "VP8L")) {
      const bits = view.getUint32(21, true);
      return ((bits & 0x3fff) + 1) * (((bits >>> 14) & 0x3fff) + 1);
    }
    if (ascii(12, "VP8X")) return ((view.getUint32(24, true) & 0xffffff) + 1) * ((view.getUint32(27, true) & 0xffffff) + 1);
    return undefined;
  }
  if (b[0] === 0xff && b[1] === 0xd8) {
    for (let at = 2; at + 9 <= b.length; ) {
      if (b[at] !== 0xff) return undefined;
      const marker = b[at + 1]!;
      if (marker === 0xff) at++;
      else if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd8)) at += 2;
      else if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return view.getUint16(at + 5) * view.getUint16(at + 7);
      } else at += 2 + view.getUint16(at + 2);
    }
  }
  return undefined;
}

/** Decoded pixels of a raster within MAX_PICTURE_PIXELS, else undefined; never SVG (unbounded cost, Office keeps a PNG). */
export function picturePixels(bytes: Uint8Array): number | undefined {
  const pixels = imagePixels(bytes);
  return pixels !== undefined && pixels <= MAX_PICTURE_PIXELS ? pixels : undefined;
}

function boxText(box: SlideBox): number {
  let n = box.caption?.length ?? 0;
  for (const p of box.paragraphs ?? []) n += p.text.length;
  for (const row of box.table ?? []) for (const cell of row) n += cell.length;
  return n;
}
const HIDDEN_SLIDE = /<(?:[\w.-]+:)?sld\b[^>]*\sshow\s*=\s*["'](?:0|false)["']/;

const MAX_SLIDE_ITEMS = 10_000;

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

function readDiagram(doc: Document, limit: number): NonNullable<SlideBox["paragraphs"]> {
  const paragraphs: NonNullable<SlideBox["paragraphs"]> = [];
  for (const pt of all(doc, "pt")) {
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

function readChart(doc: Document, budget: number): { caption?: string; table: string[][] } | null {
  const serNodes = all(doc, "ser");
  const width = Math.min(serNodes.length, MAX_CHART_SERIES) + 1;
  if (serNodes.length && width > Math.min(budget, MAX_CHART_CELLS)) return { table: [["…"]] };
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

const clip = (text: string) => (text.length > MAX_CELL_TEXT ? text.slice(0, MAX_CELL_TEXT) : text);

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

function point(xfrm: Element | undefined, name: string, a: string, b: string): [number, number] | undefined {
  const node = xfrm && children(xfrm, name)[0];
  return node ? [Number(node.getAttribute(a)) || 0, Number(node.getAttribute(b)) || 0] : undefined;
}

function readCrop(pic: Element): SlideBox["crop"] {
  const rect = first(pic, "srcRect");
  const side = (name: string) => {
    const value = Number(rect?.getAttribute(name)) / 100_000;
    return Number.isFinite(value) ? value : 0;
  };
  const [l, t, r, b] = ["l", "t", "r", "b"].map(side) as [number, number, number, number];
  if (!(l || t || r || b) || l + r >= 1 || t + b >= 1) return undefined;
  return { l, t, r, b };
}

function readFrame(shape: Element, cx: number, cy: number): SlideBox["frame"] {
  const xfrm = first(shape, "xfrm");
  const off = point(xfrm, "off", "x", "y");
  const ext = point(xfrm, "ext", "cx", "cy");
  if (!off || !ext) return undefined;
  let [x, y] = off;
  let [w, h] = ext;
  let rot = degrees(xfrm);
  let flipH = isTrue(xfrm?.getAttribute("flipH") ?? null);
  let flipV = isTrue(xfrm?.getAttribute("flipV") ?? null);
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

function degrees(xfrm: Element | undefined): number {
  return Number(xfrm?.getAttribute("rot")) / 60000 || 0;
}

export function readPptx(bytes: Uint8Array, { images = true, maxSlides = MAX_SLIDES } = {}): Deck {
  const read = archive(bytes);
  const main = mainPart(read, "ppt/presentation.xml");
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
  const pictures = new Map<string, { image: Blob; pixels: number }>();
  let pictureBytes = 0;
  let declared: ((path: string) => string | undefined) | undefined;
  const pictureType = (path: string) => imageType(path) ?? (declared ??= packageImageTypes(read))(path);
  let truncated = false;
  let textLeft = MAX_DECK_TEXT;
  for (const [index, path] of slidePaths.entries()) {
    if (!path) continue;
    if ((read.size(path) ?? 0) > MAX_XML_PART_BYTES) {
      if (slides.length === maxSlides) {
        truncated = true;
        break;
      }
      slides.push({ boxes: [{ paragraphs: [{ text: "…" }] }] });
      continue;
    }
    const part = read([path, relsPath(path)], MAX_XML_PART_BYTES);
    const slideXml = part[path];
    if (!slideXml || HIDDEN_SLIDE.test(strFromU8(slideXml.subarray(0, 16384)))) continue;
    if (slides.length === maxSlides) {
      truncated = true;
      break;
    }
    const doc = xml(part, path);
    if (!doc || ["0", "false"].includes(doc.documentElement.getAttribute("show")?.trim() ?? "")) continue;
    const slideRelList = relationshipList(part, path);
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
    let pixelsLeft = MAX_SLIDE_PIXELS;
    let cut = false;
    const keep = (box: SlideBox): boolean => {
      boxes.push(box);
      textLeft -= boxText(box);
      return textLeft <= 0 && (cut = true);
    };
    const addShape = (shape: Element): boolean => {
      const body = first(shape, "txBody");
      if (!body) return false;
      if (left <= 0) return (cut = true);
      const paragraphs: NonNullable<SlideBox["paragraphs"]> = [];
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
      return keep({
        frame: readFrame(shape, cx, cy),
        placeholder: first(shape, "ph")?.getAttribute("type") ?? (first(shape, "ph") ? "body" : undefined),
        paragraphs,
      });
    };
    const addFrame = (frame: Element): boolean => {
      if (left <= 0) return (cut = true);
      const place = () => readFrame(frame, cx, cy) ?? { x: 0.05, y: 0.25, w: 0.9, h: 0.65 };
      const chartRef = first(frame, "chart");
      const chartPath = chartRef && slideRels.get(relId(chartRef, "id") ?? "");
      const chartDoc = chartPath ? partDoc(chartPath) : null;
      const chart = chartDoc && readChart(chartDoc, left);
      if (chart) {
        left -= chart.table.reduce((n, row) => n + row.length, 0);
        if (keep({ frame: place(), ...chart })) return true;
      }
      const diagramRef = first(frame, "relIds");
      const diagramPath = diagramRef && slideRels.get(relId(diagramRef, "dm") ?? "");
      const diagramDoc = diagramPath ? partDoc(diagramPath) : null;
      const diagram = diagramDoc && left > 0 ? readDiagram(diagramDoc, left) : [];
      if (diagram.length) {
        left -= diagram.length;
        if (keep({ frame: place(), paragraphs: diagram })) return true;
      }
      const tbl = first(frame, "tbl");
      if (!tbl || left <= 0) return false;
      const table = readTable(tbl, left);
      if (!table.some((row) => row.some((cell) => cell.trim()))) return false;
      left -= table.reduce((n, row) => n + row.length, 0);
      return keep({ frame: place(), table });
    };
    const addPicture = (pic: Element): boolean => {
      const blip = first(pic, "blip");
      const target = blip && slideRels.get(relId(blip, "embed") ?? "");
      const type = target && pictureType(target);
      if (!target || !type) return false;
      if (left <= 0) return (cut = true);
      let picture = pictures.get(target);
      if (!picture) {
        if (pictureBytes + (read.size(target) ?? 0) > MAX_DECK_IMAGE_BYTES) {
          cut = true;
          return false;
        }
        const data = read([target])[target];
        if (!data) return false;
        const pixels = picturePixels(data);
        if (pixels === undefined) {
          cut = true;
          return false;
        }
        pictureBytes += data.length;
        picture = { image: new Blob([data as Uint8Array<ArrayBuffer>], { type }), pixels };
        pictures.set(target, picture);
      }
      if (picture.pixels > pixelsLeft) {
        cut = true;
        return false;
      }
      pixelsLeft -= picture.pixels;
      left--;
      boxes.push({ frame: readFrame(pic, cx, cy), image: picture.image, crop: readCrop(pic) });
      return false;
    };
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
    if (cut) boxes.push({ paragraphs: [{ text: "…" }] });
    slides.push({ boxes });
    if (textLeft <= 0) {
      truncated = index < slidePaths.length - 1;
      break;
    }
  }
  return { aspect: cy / cx, widthPt: cx / 12700, slides, truncated };
}
