// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A1111 X/Y/Z axis syntax: comma or newline lists with CSV quoting, ranges like 1-5, steps like 1-9
 * (+2), counts like 1-10 [4] (evenly spaced, truncated for ints); a descending range keeps its end
 * (5-1 (-1) is 5..1, A1111 stops at 3) and a decimal step may omit its sign.
 */

export type AxisValueType = "int" | "float" | "str";

export type ParsedValues =
  | { ok: true; values: Array<number | string> }
  | { ok: false; error: string };

// One more than A1111's own limit would never matter here: the grid is capped far below this.
const MAX_EXPANDED = 1000;

const NUM = String.raw`[+-]?\s*\d+(?:\.\d*)?`;
const INT = String.raw`[+-]?\s*\d+`;
const INT_RANGE = new RegExp(String.raw`^\s*(${INT})\s*-\s*(${INT})(?:\s*\(\s*([+-]?\s*\d+)\s*\))?\s*$`);
const INT_COUNT = new RegExp(String.raw`^\s*(${INT})\s*-\s*(${INT})\s*\[\s*(\d+)\s*\]\s*$`);
const FLOAT_RANGE = new RegExp(
  String.raw`^\s*(${NUM})\s*-\s*(${NUM})(?:\s*\(\s*([+-]?\s*\d+(?:\.\d*)?)\s*\))?\s*$`,
);
const FLOAT_COUNT = new RegExp(String.raw`^\s*(${NUM})\s*-\s*(${NUM})\s*\[\s*(\d+)\s*\]\s*$`);

// A space may follow a sign (A1111's `[+-]?\s*\d+`); anywhere else inside a number it is a typo (`1 2`, `1 e2`).
const literal = (text: string): string => text.trim().replace(/^([+-])\s+/, "$1");
const num = (text: string): number => {
  const t = literal(text);
  return /\s/.test(t) ? Number.NaN : Number(t);
};

/** Labels and requests use the same rounding as A1111's format_value (8 decimals), so 0.1+0.2 reads 0.3. */
export function roundValue(value: number): number {
  return Number(value.toFixed(8));
}

/** csv.reader(skipinitialspace=True) over every line, each field stripped; empty fields are kept here and dropped by
 *  the typed expansion, as A1111 does for numbers. */
export function splitAxisList(text: string): string[] {
  const out: string[] = [];
  let field = "";
  let quoted = false;
  let atStart = true;
  // csv.reader yields no field for a blank line, so a trailing newline adds no empty item.
  let lineHasField = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          quoted = false;
        }
      } else {
        field += ch;
      }
      continue;
    }
    if (ch === '"' && atStart) {
      lineHasField = true;
      quoted = true;
      atStart = false;
      continue;
    }
    if (ch === "," || ch === "\n" || ch === "\r") {
      if (ch === ",") lineHasField = true;
      if (ch === "," || lineHasField || field.trim()) out.push(field.trim());
      field = "";
      atStart = true;
      if (ch !== ",") lineHasField = false;
      continue;
    }
    if (atStart && (ch === " " || ch === "\t")) continue;
    atStart = false;
    field += ch;
  }
  if (lineHasField || field.trim()) out.push(field.trim());
  return out;
}

function linspace(start: number, end: number, count: number): number[] {
  if (count <= 0) return [];
  if (count === 1) return [start];
  const step = (end - start) / (count - 1);
  return Array.from({ length: count }, (_, i) => (i === count - 1 ? end : start + step * i));
}

function inclusiveRange(start: number, end: number, step: number): number[] | null {
  if (step === 0 || !Number.isFinite(step)) return null;
  if ((end - start) / step < 0) return [];
  const count = Math.floor((end - start) / step + 1e-9) + 1;
  if (count > MAX_EXPANDED) return null;
  return Array.from({ length: count }, (_, i) => start + step * i);
}

/** Parse one axis's text into typed values. `str` axes keep each list item as typed (no range expansion); empty
 *  items are dropped unless `keepEmpty`. */
export function parseAxisValues(text: string, type: AxisValueType, options: { keepEmpty?: boolean } = {}): ParsedValues {
  const items = splitAxisList(text);
  if (type === "str") {
    // Prompt S/R keeps an empty replacement (A1111 csv_string_to_list_strip drops nothing): `red, ""` compares the
    // prompt with the search text removed. Its first value is what it searches for, so that one cannot be empty.
    if (options.keepEmpty) {
      if (items.length === 0 || !items[0]) return { ok: false, error: items.some(Boolean) ? "The first value (searched for) is empty" : "No values" };
      return { ok: true, values: items };
    }
    const values = items.filter((v) => v.length > 0);
    return values.length ? { ok: true, values } : { ok: false, error: "No values" };
  }
  const values: number[] = [];
  for (const item of items) {
    if (!item) continue;
    const isInt = type === "int";
    const range = (isInt ? INT_RANGE : FLOAT_RANGE).exec(item);
    const counted = range ? null : (isInt ? INT_COUNT : FLOAT_COUNT).exec(item);
    if (range) {
      const start = num(range[1]);
      const end = num(range[2]);
      const step = range[3] !== undefined ? num(range[3]) : end >= start ? 1 : -1;
      const expanded = inclusiveRange(start, end, step);
      if (expanded === null) return { ok: false, error: `Bad range step in "${item}"` };
      values.push(...expanded.map(isInt ? Math.trunc : roundValue));
    } else if (counted) {
      const count = Number(counted[3]);
      if (count > MAX_EXPANDED) return { ok: false, error: `Too many values in "${item}"` };
      const spaced = linspace(num(counted[1]), num(counted[2]), count);
      values.push(...spaced.map(isInt ? Math.trunc : roundValue));
    } else {
      const value = num(item);
      if (!Number.isFinite(value) || !/^[+-]?\d*\.?\d+(?:e[+-]?\d+)?$|^[+-]?\d+\.$/i.test(literal(item))) {
        return { ok: false, error: `"${item}" is not a number` };
      }
      if (isInt && !Number.isInteger(value)) return { ok: false, error: `"${item}" is not a whole number` };
      values.push(isInt ? value : roundValue(value));
    }
    if (values.length > MAX_EXPANDED) return { ok: false, error: "Too many values" };
  }
  return values.length ? { ok: true, values } : { ok: false, error: "No values" };
}

/** CSV-quote a list back into the box (A1111 list_to_csv_string), for the fill button and swaps. */
export function joinAxisList(values: readonly string[]): string {
  return values.map((v) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v)).join(", ");
}
