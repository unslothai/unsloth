// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Building an Intl formatter costs ~20x a format() call, and toLocale*String builds one each time.
const dateFormats = new Map<string, Intl.DateTimeFormat>();
const numberFormats = new Map<string, Intl.NumberFormat>();

// A cached DateTimeFormat keeps its build-time zone; the OS zone can change while the page is open.
// Two UTC offsets per call, plus the zone name at most once a second for same-offset changes.
const JAN = Date.UTC(2026, 0, 1);
const JUL = Date.UTC(2026, 6, 1);
let zoneOffsets = "";
let zoneName: string | undefined;
let zoneCheckedAt = Number.NEGATIVE_INFINITY;

function dropFormatsFromOldZone(): void {
  const offsets = `${new Date(JAN).getTimezoneOffset()}/${new Date(JUL).getTimezoneOffset()}`;
  const now = Date.now();
  if (offsets === zoneOffsets && now - zoneCheckedAt < 1000 && now >= zoneCheckedAt) {
    return;
  }
  zoneOffsets = offsets;
  zoneCheckedAt = now;
  const name = new Intl.DateTimeFormat().resolvedOptions().timeZone;
  if (name !== zoneName) {
    zoneName = name;
    dateFormats.clear();
  }
}

const keyOf = (locale: string | undefined, options: object) =>
  `${locale ?? ""}\u0000${JSON.stringify(options)}`;

/** Same output as date.toLocaleString(locale, options) with explicit fields, from a cached formatter. */
export function formatDate(
  date: Date,
  locale: string | undefined,
  options: Intl.DateTimeFormatOptions,
): string {
  // format() throws on an invalid Date where toLocale*String returns "Invalid Date".
  if (Number.isNaN(date.getTime())) {
    return date.toLocaleString(locale, options);
  }
  dropFormatsFromOldZone();
  const key = keyOf(locale, options);
  let formatter = dateFormats.get(key);
  if (!formatter) {
    formatter = new Intl.DateTimeFormat(locale, options);
    dateFormats.set(key, formatter);
  }
  return formatter.format(date);
}

/** Same output as value.toLocaleString(locale, options), from a cached formatter. */
export function formatNumber(
  value: number,
  locale: string | undefined,
  options: Intl.NumberFormatOptions,
): string {
  const key = keyOf(locale, options);
  let formatter = numberFormats.get(key);
  if (!formatter) {
    formatter = new Intl.NumberFormat(locale, options);
    numberFormats.set(key, formatter);
  }
  return formatter.format(value);
}
